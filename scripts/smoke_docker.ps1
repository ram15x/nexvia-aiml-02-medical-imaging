# Build the stack, start it, and check that it works end to end. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/smoke_docker.ps1
# It needs Docker Desktop running and demo/chest_xray_example.png (run `dvc pull` if it is missing).
$ErrorActionPreference = "Stop"
$failures = @()

function Get-EnvValue($name, $default) {
    if (Test-Path .env) {
        $line = Select-String -Path .env -Pattern "^\s*$name=(.*)$" | Select-Object -First 1
        if ($line) { return $line.Matches[0].Groups[1].Value.Trim() }
    }
    return $default
}
function Check($label, [bool]$ok, $detail = "") {
    if ($ok) { Write-Host "PASS  $label $detail" -ForegroundColor Green }
    else { Write-Host "FAIL  $label $detail" -ForegroundColor Red; $script:failures += $label }
}

$apiPort = Get-EnvValue "NEXVIA_API_PORT" "8000"
$webPort = Get-EnvValue "NEXVIA_WEB_PORT" "5173"
$image = "demo/chest_xray_example.png"
if (-not (Test-Path $image)) { throw "Missing $image. Run: dvc pull demo/chest_xray_example.png.dvc" }

try {
    docker compose up -d --build --wait
    Check "stack started and the API healthcheck passed" ($LASTEXITCODE -eq 0)

    $health = curl.exe -s "http://127.0.0.1:$apiPort/health" | ConvertFrom-Json
    Check "/health reports a checkpoint" ($health.model_available -eq $true) "(status: $($health.status))"
    Check "spleen checkpoint is available" ($health.segmentation.status -eq "available")

    $predict = curl.exe -s -F "file=@$image;type=image/png" "http://127.0.0.1:$apiPort/predict" | ConvertFrom-Json
    Check "/predict returns 14 finding scores" ($predict.findings.Count -eq 14) "(got $($predict.findings.Count))"

    $web = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:$webPort/"
    Check "web UI answers on port $webPort" ($web -eq "200") "(HTTP $web)"

    $uid = (docker compose exec -T api id -u).Trim()
    Check "API container runs as a non-root user" ($uid -ne "0") "(uid $uid)"

    $leaked = docker compose exec -T api sh -c "ls /app/.env 2>/dev/null; ls /app/data 2>/dev/null"
    Check "no .env or data folder inside the API image" ([string]::IsNullOrWhiteSpace($leaked))
}
finally {
    if ($failures.Count -gt 0) { docker compose logs --tail 60 api }
    docker compose down
}
if ($failures.Count -gt 0) { Write-Host "`n$($failures.Count) check(s) failed." -ForegroundColor Red; exit 1 }
Write-Host "`nAll checks passed." -ForegroundColor Green
