**avgbpm.py
Calculates the BPM of all tracks in current folder and displays BPM and total average BPM on a random person's computer 20 years in the future.**
pip install librosa
winget install Gyan.FFmpeg
python 3.11 
python avgbpm.py
 

**generate_lyrics.py
transcribes lyrics and generates timestamped .srt files**
pip install faster-whisper
(I had to download some cuda DLLs for ctranslate2 to get this to work.  Should be ok if you're on CUDA 12)
python generate_lyrics.py Filename 
options:
--out  subtitlefilename                     lets you name the subtitle file however you want, in this case it would be called subtitlefilename.srt
--folder "C:\Music\Album1\"                 automatically generates subtitles for every track in the folder, auto naming them

**wav2mp3.ps1 
Powershell script to convert .wav to 320kbps mp3.  Useful when your hard drive is full up.  Contains a validation check to ensure the output file is within 5 secs of the original and fails on any error.  Removes .wav file afterwards.
