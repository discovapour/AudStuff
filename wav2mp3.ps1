<#
.SYNOPSIS
  Convert WAV -> 320 kbps CBR MP3, validate, then remove the WAV.
.EXAMPLE
  .\wav2mp3.ps1 -Path D:\Audio -Recurse
  .\wav2mp3.ps1 -Path D:\Audio -Permanent -WhatIf
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string]$Path,
    [switch]$Recurse,
    [switch]$Permanent,          # default: send WAV to Recycle Bin
    [double]$Tolerance = 0.25    # max allowed duration drift (seconds)
)

foreach ($t in 'ffmpeg', 'ffprobe') {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t not found in PATH" }
}
Add-Type -AssemblyName Microsoft.VisualBasic

function Get-AudioInfo([string]$File) {
    $j = & ffprobe -v error -select_streams a:0 `
        -show_entries format=duration:stream=sample_rate,channels -of json $File | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $j.streams) { return $null }
    [pscustomobject]@{
        Duration   = [double]$j.format.duration
        SampleRate = [int]$j.streams[0].sample_rate
        Channels   = [int]$j.streams[0].channels
    }
}

$ok = 0; $fail = 0; $skip = 0
$files = Get-ChildItem -LiteralPath $Path -Filter *.wav -File -Recurse:$Recurse

foreach ($f in $files) {
    $wav = $f.FullName
    $mp3 = [IO.Path]::ChangeExtension($wav, '.mp3')
    $tmp = "$mp3.part"

    if (Test-Path -LiteralPath $mp3) { Write-Warning "SKIP (mp3 exists): $wav"; $skip++; continue }

    $src = Get-AudioInfo $wav
    if (-not $src -or $src.Duration -le 0) { Write-Warning "FAIL (unreadable wav): $wav"; $fail++; continue }

    $ffArgs = @('-hide_banner', '-nostdin', '-v', 'error', '-y', '-i', $wav,
        '-map', '0:a:0', '-map_metadata', '0',
        '-c:a', 'libmp3lame', '-b:a', '320k', '-id3v2_version', '3')
    if ($src.SampleRate -gt 48000) { $ffArgs += '-ar', '48000' }  # MP3 max is 48 kHz
    if ($src.Channels -gt 2) { $ffArgs += '-ac', '2' }            # MP3 max is stereo
    $ffArgs += '-f', 'mp3', $tmp

    & ffmpeg @ffArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $tmp)) {
        Write-Warning "FAIL (encode): $wav"; Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue; $fail++; continue
    }

    # Validation 1: full decode with no errors
    $decodeErr = & ffmpeg -hide_banner -nostdin -v error -i $tmp -f null - 2>&1
    # Validation 2: duration + bitrate match
    $dst = Get-AudioInfo $tmp
    $br  = [int](& ffprobe -v error -select_streams a:0 -show_entries stream=bit_rate -of default=nw=1:nk=1 $tmp)
    $drift = if ($dst) { [math]::Abs($dst.Duration - $src.Duration) } else { [double]::MaxValue }

    $problems = @()
    if ($decodeErr)             { $problems += "decode errors" }
    if (-not $dst)              { $problems += "unreadable mp3" }
    if ($drift -gt $Tolerance)  { $problems += ("duration drift {0:N3}s" -f $drift) }
    if ($br -lt 319000)         { $problems += "bitrate $br" }

    if ($problems) {
        Write-Warning "FAIL ($($problems -join ', ')): $wav"
        Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
        $fail++; continue
    }

    Move-Item -LiteralPath $tmp -Destination $mp3
    if ($PSCmdlet.ShouldProcess($wav, 'Delete WAV')) {
        if ($Permanent) { Remove-Item -LiteralPath $wav }
        else { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($wav, 'OnlyErrorDialogs', 'SendToRecycleBin') }
    }
    Write-Host ("OK  {0}  (drift {1:N3}s)" -f $f.Name, $drift)
    $ok++
}

Write-Host "`nDone. OK: $ok  Failed: $fail  Skipped: $skip"
if ($fail) { exit 1 }
