Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/c', 'C:\Users\Ayaka\RetNet\start_v12.bat' `
    -WorkingDirectory 'C:\Users\Ayaka\RetNet' `
    -WindowStyle Hidden
Write-Output "launched"
