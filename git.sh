# chmod +x git.sh
# 1.将所有的项目代码文件上传到暂存区
git add .
# 2.查看当前git的状态
git status
# 3.将暂存区的文件进行commit
git commit -m "add some test code 20250903"
git remote -v
# ４.把本地仓库的变化连接到远程仓库主分支
git pull origin main --allow-unrelated-histories
# 5.使用push指令进行上传,
git push origin main


# git pull origin main
