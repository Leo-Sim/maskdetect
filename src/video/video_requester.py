
import yt_dlp

video_url = 'https://www.youtube.com/watch?v=eurAOZuzTag'

ydl_opts = {
    'outtmpl': 'video/%(title)s.%(ext)s',
    'format': 'bestvideo'
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=eurAOZuzTag'])