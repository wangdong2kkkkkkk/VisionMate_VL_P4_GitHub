# VisionMate VL-P1 — 25 photos from Wikimedia Commons
# Chạy từ PowerShell Windows 5.1 sau khi giải nén vào research\vl_p1\
# Chỉ tải về data\originals, KHÔNG tạo train/test hay chỉnh sửa production.
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$manifestPath = Join-Path $PSScriptRoot 'photo_sources.json'
$records = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$saveRoot = Join-Path $PSScriptRoot 'data\originals'
$sourceLog = Join-Path $PSScriptRoot 'downloaded_photo_sources.csv'
$credits = New-Object System.Collections.Generic.List[object]
$UserAgent = 'VisionMateVL-P1/0.1 (non-commercial educational research; Wikimedia Commons attribution retained)'
$ok = 0; $failed = 0
foreach ($item in $records) {
    $folder = Join-Path $saveRoot $item.object_id
    if (-not (Test-Path $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
    $targetName = ('{0:D2}.jpg' -f [int]$item.photo_number)
    $target = Join-Path $folder $targetName
    $title = 'File:' + $item.commons_filename
    Write-Host ("[{0}] #{1} {2}" -f $item.object_id, $item.photo_number, $title)
    if ($DryRun) { continue }
    try {
        $api = 'https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url%7Cextmetadata%7Cmime&iiurlwidth=1280&titles=' + [Uri]::EscapeDataString($title)
        $res = Invoke-RestMethod -Uri $api -UserAgent $UserAgent -TimeoutSec 45
        $p = @($res.query.pages.PSObject.Properties | ForEach-Object { $_.Value })[0]
        if ($null -eq $p.imageinfo -or $p.imageinfo.Count -eq 0) { throw 'No Commons image info (file may be missing)' }
        $info = $p.imageinfo[0]
        if ($info.mime -ne 'image/jpeg') { throw ('Expected JPEG; got ' + $info.mime) }
        $src = if ($info.thumburl) { $info.thumburl } else { $info.url }
        if (-not (Test-Path -LiteralPath $target)) {
            Invoke-WebRequest -Uri $src -OutFile $target -UseBasicParsing -UserAgent $UserAgent -TimeoutSec 90
        }
        $license = if ($info.extmetadata.LicenseShortName) { [string]$info.extmetadata.LicenseShortName.value } else { 'CHECK_MANUALLY' }
        $artistHtml = if ($info.extmetadata.Artist) { [string]$info.extmetadata.Artist.value } else { '' }
        $artist = [Net.WebUtility]::HtmlDecode(([regex]::Replace($artistHtml, '<[^>]*>', ' '))) -replace '\s+', ' '
        $credits.Add([pscustomobject]@{
          object_id=$item.object_id;label_vi=$item.label_vi;photo_number=$item.photo_number
          local_path=$target;source_page=$item.source_page;license=$license;author=$artist.Trim()
          sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
        })
        $ok++
        Write-Host ('  Saved ' + $targetName + ' | ' + $license) -ForegroundColor Green
    } catch {
        $failed++
        Write-Warning ("  Failed: {0}. Open the source page and download manually: {1}" -f $_.Exception.Message, $item.source_page)
    }
    Start-Sleep -Milliseconds 300
}
if (-not $DryRun) {
  $credits | Export-Csv -LiteralPath $sourceLog -NoTypeInformation -Encoding UTF8
  Write-Host ("Finished: {0} files downloaded or verified; {1} failed. Credits: {2}" -f $ok, $failed, $sourceLog)
  Write-Host 'IMPORTANT: review each photo, license and class before cropping for train; do NOT use the same source photo in test.'
}
