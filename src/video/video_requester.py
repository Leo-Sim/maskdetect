
import yt_dlp

video_url = 'https://www.youtube.com/watch?v=wxmmrwYT2SE'

ydl_opts = {
    'outtmpl': 'video/%(title)s.%(ext)s',
    'format': 'best[height<=480]'
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=wxmmrwYT2SE'])