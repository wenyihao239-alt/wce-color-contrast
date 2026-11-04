# GitHub upload steps

This folder is a ready-to-push repository.

`powershell
cd C:\wce\upload_github\wce-color-contrast
git status
git add .
git commit -m "Initial commit: WCE colour-contrast pilot (V1/V2/V2.1)"
`

Create an empty public repository on GitHub, then:

`powershell
git remote add origin https://github.com/<YOUR_USER>/<REPO>.git
git branch -M main
git push -u origin main
`

If you use GitHub CLI:

`powershell
gh repo create <REPO> --public --source=. --remote=origin --push
`

Notes:
- Raw images, derived condition images and model weights are **not** in this repo.
- Put them on Google Drive using the prepared upload_gdrive folder.
- 	hird_party/WCE2024 is not committed; scripts/setup_third_party.sh clones it and applies patches/wce2024_yolov5_compat.patch.
- Check LICENSE, NOTICE.md, DATA.md before publishing.
