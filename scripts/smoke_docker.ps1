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

# Part 2: the single-container image used for hosted deployments (API plus UI on one port).
$hostedPort = 17860
try {
    Write-Host "`nBuilding the hosted single-container image..."
    docker build -f deploy/hf-space/Dockerfile -t nexvia-hosted-smoke .
    Check "hosted image built" ($LASTEXITCODE -eq 0)
    docker rm -f nexvia-hosted-smoke 2>$null | Out-Null
    docker run -d --name nexvia-hosted-smoke -p "127.0.0.1:${hostedPort}:7860" nexvia-hosted-smoke | Out-Null
    $up = $false
    for ($i = 0; $i -lt 40 -and -not $up; $i++) {
        Start-Sleep -Seconds 3
        $code = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:$hostedPort/health"
        if ($code -eq "200") { $up = $true }
    }
    Check "hosted container answers /health" $up
    if ($up) {
        $h = curl.exe -s "http://127.0.0.1:$hostedPort/health" | ConvertFrom-Json
        Check "hosted: Orthanc is reported as disabled" ($h.orthanc.enabled -eq $false)
        $p = curl.exe -s -F "file=@$image;type=image/png" "http://127.0.0.1:$hostedPort/predict" | ConvertFrom-Json
        Check "hosted: /predict returns 14 scores" ($p.findings.Count -eq 14)
        $ui = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:$hostedPort/"
        Check "hosted: the UI is served on /" ($ui -eq "200") "(HTTP $ui)"
        $orth = curl.exe -s -o NUL -w "%{http_code}" -X POST "http://127.0.0.1:$hostedPort/orthanc/sync"
        Check "hosted: Orthanc routes answer 404" ($orth -eq "404") "(HTTP $orth)"
        $huid = (docker exec nexvia-hosted-smoke id -u).Trim()
        Check "hosted: container runs as a non-root user" ($huid -ne "0") "(uid $huid)"
    }
}
finally {
    if ($failures.Count -gt 0) { docker logs --tail 60 nexvia-hosted-smoke 2>$null }
    docker rm -f nexvia-hosted-smoke 2>$null | Out-Null
}
if ($failures.Count -gt 0) { Write-Host "`n$($failures.Count) check(s) failed." -ForegroundColor Red; exit 1 }
Write-Host "`nAll checks passed." -ForegroundColor Green
