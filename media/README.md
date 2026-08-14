# Blog media

Photos and videos for the article at [`../index.html`](../index.html)
(published at https://lpslv.github.io/openpi/).

Drop files here, then open `index.html`, find the matching
`MEDIA PLACEHOLDER` block, and delete the two comment marker lines around it.
Each block is already written with the right paths, `alt` text and caption —
nothing else needs editing.

## Slots the article is waiting for

| File | Where it appears | What it should show |
|---|---|---|
| `hero.jpg` | lead image, under the abstract | The rig mid-trial — the first thing a reader sees |
| `setup-workspace.jpg` | §1, full width | The whole rig: UR5e, Hand-E gripper, tabletop, drop box, shoulder camera at the back-left |
| `setup-gripper.jpg` | §7, left of a pair | Close-up of the Hand-E gripper holding the blue block |
| `setup-wrist-view.jpg` | §7, right of a pair | A frame from the wrist camera mid-grasp |
| `policy-run.mp4` | §6, full width | The step-150 checkpoint completing a trial in distribution |
| `policy-run-poster.jpg` | poster frame for the above | A representative still from the video |

Add more slots by copying any existing `<figure>` block in `index.html`.

## Specs

**Photos.** JPEG, 1600 px on the long edge, quality ~82, under about 400 KB
each. That is enough for a 736 px column on a 2× display. Strip EXIF if the
photos carry GPS. Landscape orientation matches the layout best.

```bash
# resize and compress, requires ImageMagick
magick input.jpg -auto-orient -resize 1600x1600\> -strip -quality 82 setup-workspace.jpg
```

**Videos.** H.264 MP4, 1280×720, no audio track (nobody wants sound from a
robot video), target under 8 MB. Keep clips to 10–20 seconds — they illustrate
a point, they are not documentation.

```bash
# transcode, trim, drop audio, generate a poster frame
ffmpeg -i raw.mov -t 20 -an -vf "scale=1280:-2" -c:v libx264 -crf 26 \
       -preset slow -movflags +faststart policy-run.mp4
ffmpeg -i policy-run.mp4 -ss 2 -vframes 1 -q:v 3 \
       policy-run-poster.jpg
```

`-movflags +faststart` matters: without it the browser downloads the whole file
before the first frame appears.

## Before you commit large files

Git keeps every version of a binary forever, and GitHub Pages has real limits:

- **100 MB** hard limit per file
- **1 GB** recommended limit for the whole published site
- **100 GB/month** soft bandwidth limit

A handful of compressed clips is fine. If you end up wanting many videos, or
anything longer than about 30 seconds, host them on YouTube or Vimeo and embed
instead — an unlisted YouTube video works and costs the repo nothing. Replace
the `<video>` block with an `<iframe>` if you go that way.

Check what you are about to add:

```bash
du -sh media/
git add -n media/          # dry run, lists what would be staged
```
