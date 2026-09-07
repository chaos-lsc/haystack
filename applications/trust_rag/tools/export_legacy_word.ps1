param(
    [Parameter(Mandatory = $true)][string]$SourceDirectory,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$destination = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
$word.Options.UpdateLinksAtOpen = $false
try {
    foreach ($source in Get-ChildItem -LiteralPath $SourceDirectory -Filter *.doc -File | Sort-Object Name) {
        $digest = (Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $target = Join-Path $destination ($digest + '.json')
        if (Test-Path -LiteralPath $target) { continue }
        $document = $null
        try {
            # Read-only, no recent-file entry, macros and link updates disabled.
            $document = $word.Documents.Open($source.FullName, $false, $true, $false)
            $content = $document.Content.Text
            if ([string]::IsNullOrWhiteSpace($content)) { throw "Empty Word document: $($source.Name)" }
            $record = [ordered]@{
                source_path = $source.FullName
                file_name = $source.Name
                source_sha256 = $digest
                converter = 'Microsoft Word COM Content.Text; read-only'
                text = $content
            }
            $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $target -Encoding utf8
            Write-Output "$($source.Name): $($content.Length) characters"
        } finally {
            if ($null -ne $document) {
                $document.Close(0)
                [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
            }
        }
    }
} finally {
    $word.Quit(0)
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
}
