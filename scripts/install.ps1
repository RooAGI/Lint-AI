param(
    [ValidateSet('', 'codex', 'claude', 'gemini', 'agy')]
    [string]$Agent = '',
    [string]$Project = '.',
    [string]$InstallDir = "$HOME\.local\bin"
)

$ErrorActionPreference = 'Stop'
$Repo = 'RooAGI/Membrick'
$BaseUrl = "https://github.com/$Repo/releases/latest/download"
$Asset = 'membrick-windows-x86_64.exe'
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("membrick-" + [guid]::NewGuid().ToString('N'))
$BinaryTemp = Join-Path $TempDir $Asset
$ChecksumTemp = "$BinaryTemp.sha256"
$InstalledBinary = Join-Path $InstallDir 'membrick.exe'

New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
try {
    Write-Host 'Downloading official Membrick release for Windows x86_64...'
    Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/$Asset" -OutFile $BinaryTemp
    Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/$Asset.sha256" -OutFile $ChecksumTemp

    $Expected = ((Get-Content -Raw $ChecksumTemp).Trim() -split '\s+')[0].ToLowerInvariant()
    $Actual = (Get-FileHash -Algorithm SHA256 $BinaryTemp).Hash.ToLowerInvariant()
    if ($Expected -ne $Actual) {
        throw 'Checksum verification failed; refusing to install.'
    }

    Write-Host 'Checksum verified.'
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item -Force $BinaryTemp $InstalledBinary

    $Version = & $InstalledBinary --version
    Write-Host "Installed $Version to $InstalledBinary"

    $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $PathEntries = @($UserPath -split ';' | Where-Object { $_ })
    if ($PathEntries -notcontains $InstallDir) {
        $NewUserPath = if ([string]::IsNullOrWhiteSpace($UserPath)) { $InstallDir } else { "$UserPath;$InstallDir" }
        [Environment]::SetEnvironmentVariable('Path', $NewUserPath, 'User')
        Write-Host "Added $InstallDir to your user PATH. Open a new terminal to use membrick by name."
    }
    if (($env:Path -split ';') -notcontains $InstallDir) {
        $env:Path = "$InstallDir;$env:Path"
    }

    if ($Agent) {
        switch ($Agent) {
            'codex' {
                $InstallFlag = '--codex-install'
                $VerifyFlag = '--codex-verify-mcp'
            }
            'claude' {
                $InstallFlag = '--claude-code-install'
                $VerifyFlag = '--claude-code-verify-mcp'
            }
            'gemini' {
                $InstallFlag = '--gemini-cli-install'
                $VerifyFlag = '--gemini-cli-verify-mcp'
            }
            'agy' {
                $InstallFlag = '--agy-install'
                $VerifyFlag = '--agy-verify-mcp'
            }
        }

        Write-Host "Configuring Membrick for $Agent in $Project..."
        & $InstalledBinary $InstallFlag $Project
        if ($LASTEXITCODE -ne 0) { throw "Membrick $Agent configuration failed." }

        Write-Host 'Verifying the Membrick MCP runtime...'
        & $InstalledBinary $VerifyFlag $Project
        if ($LASTEXITCODE -ne 0) { throw 'Membrick MCP verification failed.' }

        Write-Host "Membrick is installed, configured for $Agent, and MCP verification passed."
    } else {
        Write-Host 'Membrick is installed. Re-run with -Agent codex|claude|gemini|agy to configure an agent.'
    }
}
finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $TempDir
}
