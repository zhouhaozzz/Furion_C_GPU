@echo off
    
REM 进入bin文件夹
cd bin/

REM 删除bin\data、bin\python_plot文件夹
set datafolder="data"
if exist "%datafolder%" (
    rmdir /s /q "%datafolder%"
    echo Folder "%datafolder%" deleted.
) else (
    echo Folder "%datafolder%" not found.
)  

set plotfolder="python_plot"
if exist "%plotfolder%" (
    rmdir /s /q "%plotfolder%"
    echo Folder "%plotfolder%" deleted.
) else (
    echo Folder "%plotfolder%" not found.
)  

REM 创建bin\data 文件夹
mkdir data
echo Folder "data" created.

REM 复制python_plot文件夹到bin文件夹
set sourceFolder="..\src\python_plot" 
set destinationFolder="python_plot"

xcopy /s /i %sourceFolder% %destinationFolder%

echo Folder "python_plot" copy.


echo The initialization of bin is complete...
echo Start run...

REM 可执行文件的名称或路径
set executable=Furion.exe 

REM 要启动的进程数量
REM set num_processes=10

REM 构建 mpiexec 命令
REM set mpi_command=mpiexec -n %num_processes% %executable%

set mpi_command= %executable%

REM 执行 mpiexec 命令
%mpi_command%

REM 检查错误代码并提供适当的消息
if %errorlevel% neq 0 (
    echo Program execution error.
) else (
    echo Program executed successfully.
)

pause  REM 暂停以查看输出，按任意键继续
