## what is Git?
### 答:
版本控制工具 GitHub的用户端 方便团队协作

## 常用Git指令
### 与文件夹建立关联
>使用git打开文件夹
```bash
git init
```
### 添加到暂存区
```bash
git add <文件>
git add .
```
### 配置/查看个人信息
```bash
git config --global user.email "邮箱"
git config --global user.name "名字"
```
> 配置文件位于 git安装目录里面
> 只针对本项目的话将--global 换成--local
> 这样的话相关信息会被放在 /.git/config下

### 忽略文件
> .gitignore用于设置什么后缀的文件会被git忽略掉

### 查看状态
```bash
git status
```
> 红色的是修改/添加了但未添加到暂存区的文件
> 绿色的是已经添加到暂存区的文件

### 添加到版本库
```bash
git commit -m "描述"
```

### 查看日志/回退版本
```bash
git log
git reflog
git reset --hard 版本号
git reset --mixed 版本号
git reset --soft 版本号
```
> 版本号即 git log打开的日志中显示的hash值  
> hash值对应commit时刻状态 
> mixed 清空暂存区 保留工作区  
> hard 清空暂存区和工作区  
> soft 保留暂存区 保留工作区  
- 注意:git判断repo和本地哪个版本更领先的机制是看commit结点
- 可以使用hard避免强制push情况 代价是损失一些进度

### 分支
- 查看分支
```bash
git branch
```
- 创建分支
```bash
git branch 分支名 
```
- 切换分支
```bash
git checkout 分支名 
```
- 合并分支
```bash
git merge 
```
- 删除分支
```bash
git branch -d 分支名 
```
### 远程仓库
- 添加仓库
```bash
git remote add origin 仓库URL 
```
- 推送工作区
```bash
git push -u origin 分支名 
```
- 克隆仓库
```bash
git clone 仓库URL 
```
- 拉取更改
```bash
git pull origin 分支名 
```


