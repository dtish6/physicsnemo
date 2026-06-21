# Lightweight temperature watchdog.
# Logs GPU (and best-effort CPU) temperature to a CSV every IntervalSec seconds.
# Each line is flushed to disk immediately, so the log survives even if the
# machine is shut down / the process is killed -- you keep everything up to the
# last sample.
#
# Start it:
#   powershell -ExecutionPolicy Bypass -File temp_watchdog.ps1
# Read it in the morning (quick summary):
#   Import-Csv temp_watchdog.csv | Measure-Object gpu_temp_C -Maximum -Minimum -Average

param(
    [int]$IntervalSec = 60,
    [string]$LogPath  = "D:\Nemo\physicsnemo\examples\elasticity_3d\temp_watchdog.csv"
)

if (-not (Test-Path $LogPath)) {
    "timestamp,gpu_temp_C,gpu_util_pct,gpu_power_W,gpu_mem_used_MiB,cpu_temp_C" |
        Out-File -FilePath $LogPath -Encoding utf8
}

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # --- GPU via nvidia-smi (reliable) ---
    $gtemp = "NA"; $gutil = "NA"; $gpow = "NA"; $gmem = "NA"
    try {
        $g = & nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used `
                          --format=csv,noheader,nounits
        $parts = ($g -split ",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 4) {
            $gtemp = $parts[0]; $gutil = $parts[1]; $gpow = $parts[2]; $gmem = $parts[3]
        }
    } catch { }

    # --- CPU thermal zone (best effort; many laptops don't expose it) ---
    $ctemp = "NA"
    try {
        $tz = Get-CimInstance -Namespace "root/wmi" -ClassName MSAcpi_ThermalZoneTemperature `
                              -ErrorAction Stop | Select-Object -First 1
        if ($tz -and $tz.CurrentTemperature -gt 0) {
            $ctemp = [math]::Round(($tz.CurrentTemperature / 10) - 273.15, 1)
        }
    } catch { }

    "$ts,$gtemp,$gutil,$gpow,$gmem,$ctemp" | Out-File -FilePath $LogPath -Append -Encoding utf8
    Start-Sleep -Seconds $IntervalSec
}
