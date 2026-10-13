Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/c', 'C:\Users\Ayaka\RetNet\start_v13.bat' `
    -WorkingDirectory 'C:\Users\Ayaka\RetNet' `
    -WindowStyle Hidden
Write-Output "launched"
