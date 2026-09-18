Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
shell.Run """" & dir & "\.venv\Scripts\pythonw.exe"" """ & dir & "\app.py""", 0, False
