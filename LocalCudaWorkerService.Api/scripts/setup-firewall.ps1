param(
    [int]$Port = 32141,
    [string]$RuleNameIn = "LocalCudaWorkerService_Inbound",
    [string]$RuleNameOut = "LocalCudaWorkerService_Outbound",
    [switch]$SystemEnv
)

function Assert-Admin {
    $wid = [Security.Principal.WindowsIdentity]::GetCurrent()
    $prp = New-Object Security.Principal.WindowsPrincipal($wid)
    if (-not $prp.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "[i] Starte neu mit Admin-Rechten..."
        Start-Process powershell "-ExecutionPolicy Bypass -File `"$PSCommandPath`" $($MyInvocation.UnboundArguments)" -Verb RunAs
        exit
    }
}

Assert-Admin

Write-Host "== Firewall-Regeln für Port $Port =="

# Inbound
$existsIn = (Get-NetFirewallRule -DisplayName $RuleNameIn -ErrorAction SilentlyContinue)
if (-not $existsIn) {
    New-NetFirewallRule -DisplayName $RuleNameIn -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
    Write-Host "[OK] Eingehende Regel erstellt."
} else {
    Set-NetFirewallRule -DisplayName $RuleNameIn -Enabled True
    Write-Host "[OK] Eingehende Regel existiert bereits."
}

# Outbound (meist nicht nötig, optional)
$existsOut = (Get-NetFirewallRule -DisplayName $RuleNameOut -ErrorAction SilentlyContinue)
if (-not $existsOut) {
    New-NetFirewallRule -DisplayName $RuleNameOut -Direction Outbound -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
    Write-Host "[OK] Ausgehende Regel erstellt."
} else {
    Set-NetFirewallRule -DisplayName $RuleNameOut -Enabled True
    Write-Host "[OK] Ausgehende Regel existiert bereits."
}

# Env Variable setzen
$envVarName = "ASPNETCORE_URLS"
$envValue = "https://0.0.0.0:$Port"

if ($SystemEnv) {
    [Environment]::SetEnvironmentVariable($envVarName, $envValue, [EnvironmentVariableTarget]::Machine)
    Write-Host "[OK] Systemweite Variable $envVarName = $envValue"
} else {
    [Environment]::SetEnvironmentVariable($envVarName, $envValue, [EnvironmentVariableTarget]::User)
    Write-Host "[OK] Benutzer-Variable $envVarName = $envValue"
}

Write-Host "Starte Test (falls Dienst läuft)..."
try {
    $res = Invoke-WebRequest -Uri "https://localhost:$Port/swagger/v1/swagger.json" -UseBasicParsing -SkipCertificateCheck -TimeoutSec 5
    Write-Host "[TEST] Lokal erreichbar: HTTP $($res.StatusCode)"
} catch {
    Write-Host "[WARN] Lokaler Test fehlgeschlagen: $($_.Exception.Message)"
}

Write-Host "Fertig. Bitte neue Shell öffnen, damit Env greift."