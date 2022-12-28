from __future__ import unicode_literals
import youtube_dl


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')
    if d['status'] == 'downloading':
        p = d['_percent_str']
        p = p.replace('%','')
        print(p)
        print(d['filename'], d['_percent_str'], d['_eta_str'])


ydl_opts = {
    'logger': MyLogger(),
    'progress_hooks': [my_hook],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=4-zPw6OuRqE'])