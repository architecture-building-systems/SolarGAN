@REM Script to build and create installer

@REM Base directorie
set SOLARGAN_BUILD_DIR=%~dp0
set GIT_DIR=%SOLARGAN_BUILD_DIR%..\..
SET SOLARGAN_DIR=%SOLARGAN_BUILD_DIR%..

@REM Change these paths accordingly
SET MSBUILD=%PROGRAMFILES(x86)%\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe
SET MAKENSIS=%PROGRAMFILES(X86)%\NSIS\makensis.exe

CD %HIVE_BUILD_DIR%

@REM Build the SolarGAN geometry exporter
echo Building SolarGAN geometry exporter...
cd %SOLARGAN_DIR%\geometry_exporter\SolarGAN\Hive.IO\
"%MSBUILD%" SolarGAN.csproj /p:PreBuildEvent="" /p:PostBuildEvent=""

echo ...Done

@REM Build Installer
echo Building Installer Setup_SolarGAN.exe...
cd %SOLARGAN_BUILD_DIR%
"%MAKENSIS%" solargan.nsi
echo ...Done

cd %SOLARGAN_BUILD_DIR%
echo ...Done

pause