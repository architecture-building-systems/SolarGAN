# NSIS script for creating the SolarGAN installer
SetCompressor /FINAL lzma

; include logic library
!include 'LogicLib.nsh'

; include the modern UI stuff
!include "MUI2.nsh"

# icon stuff
#!define MUI_ICON "hive.ico"

Name "SolarGAN"

!define MUI_FILE "savefile"
!define MUI_BRANDINGTEXT "SolarGAN"
CRCCheck On


OutFile "Setup_SolarGAN.exe"


;--------------------------------
;Folder selection page

InstallDir "$APPDATA\Grasshopper\Libraries\SolarGAN"

;Request application privileges for Windows Vista
RequestExecutionLevel user

;--------------------------------
;Interface Settings

!define MUI_ABORTWARNING

;--------------------------------
;Pages

!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
;Languages

  !insertmacro MUI_LANGUAGE "English"

;--------------------------------
;Installer Sections

Section "SolarGAN" Base_Installation_Section
    SectionIn RO  # this section is required
    SetOutPath "$INSTDIR"

    # first, delete any "old" files (some of them have been renamed...)

    Delete /REBOOTOK "$INSTDIR\SolarGAN.gha"
    Delete /REBOOTOK "$INSTDIR\octree.oct"
    Delete /REBOOTOK "$INSTDIR\output.tif"
    Delete /REBOOTOK "$INSTDIR\sky.rad"
    Delete /REBOOTOK "$INSTDIR\sky_overcast.mat"
    Delete /REBOOTOK "$INSTDIR\test.rad"

    # next, copy the "new files"

    # SolarGAN and dependencies
    File /oname=SolarGAN.gha "..\geometry_exporter\SolarGAN\bin\Debug\net48\SolarGAN.gha"  
    File ".\radiance\octree.oct"
    File ".\radiance\sky.rad"
    File ".\radiance\sky_overcast.mat"
    File ".\radiance\test.rad"

SectionEnd

;Section "Radiance" Radiance_Installation_Section

; SectionIn RO

; IfFileExists "$INSTDIR\..\ProvingGround.Conduit.gha" 0 file_not_found
;     goto end_of_block
;     file_not_found:
;     File "ProvingGround.Conduit.gha"
;     end_of_block:

;SectionEnd

LangString DESC_Section1 ${LANG_ENGLISH} "Installs the SolarGAN plugin for Grasshopper."
;LangString DESC_Section2 ${LANG_ENGLISH} "Install Radiance. Visit https://apps.proving$\nground.io/conduit/ for more information."

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${Base_Installation_Section} $(DESC_Section1)
  ;!insertmacro MUI_DESCRIPTION_TEXT ${Conduit_Installation_Section} $(DESC_Section2)
!insertmacro MUI_FUNCTION_DESCRIPTION_END