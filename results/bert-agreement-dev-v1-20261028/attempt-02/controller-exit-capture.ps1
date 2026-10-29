# Excerpt of the local wrapper immediately after the controller returned.
# The launch preamble and local absolute paths are not retained in this file.
$controllerExit=$LASTEXITCODE; if (Test-Path -LiteralPath $output) { [IO.File]::WriteAllText((Join-Path $output 'controller-exit.json'), (([ordered]@{schema='bert-agreement-wrapper-exit-v1'; exit_code=$controllerExit} | ConvertTo-Json -Compress) + "`n"), [Text.UTF8Encoding]::new($false)) }; Write-Output "Controller process exit: $controllerExit"; exit $controllerExit
