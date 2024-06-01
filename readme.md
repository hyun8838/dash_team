# Virtual Environment

### `venv`
```{python}
### venv
!python -m venv NAME
!NAME\Scripts\activate
!pip install -r requirements.txt

# 주피터 노트북에서 가상환경 사용하려면, 주피터 노트북 내부에 가상환경을 설치해야 함.
!pip install jupyter
!pip install ipykernel
!python -m ipykernel install --user --name NAME --display-name "출력될 커널 이름"

# requirements.txt 생성
!dir /B Lib\site-packages
!pip freeze > requirements.txt

!deactivate
```
### `conda`
```{python}
### conda
!conda create --name NAME
!conda create --name NAME python=VERSION
# conda는 venv와는 달리 가상 환경을 현재 폴더에 생성하지 않고 아나콘다 설치 폴더의 envs 안에 생성함.
!activate NAME
!conda install PACKAGE.NAME
!conda search PACKAGE.NAME

# conda yaml 파일로 새로운 가상환경 만들때
! conda env update --file environment.yml -- 가상환경 activate 되어있을때
! conda env update --name 내가만들 가상환경이름 --file environment.yml

# requirements.txt 생성
!conda list --export > package-list.txt
!conda install --file package-list.txt
```

### 아마 여기서 막힐거 같은데... conda 없이 pip 만으로 깔고 싶으면?

그래서 `compatible_requirements.txt` 만들어놨음.

`!pip install -r compatible_requirements.txt` 하면 되는데, 이때 pip 를 python 3.12 로 해야함

근데 일일히 다운받아야 할 수 있음. 콘다랑 호환이 안되는 패키지가 있을수 있기 때문에...

그래서 추천하는건 그냥 콘다 깔고 쓰는게 편함.


## How to use `git`
```
### 설정
$ git config --global user.name "MY NAME"
$ git config --global user.email "junior0101@naver.com"

$ git init
# git remote add origin https://github.com/YoungnohLee/etc.git # git 을 원격 저장소에 저장하게 해주는 엔드포인트

$ git pull https://github.com/YoungnohLee/etc.git # git 원격 저장소와 동기화
# git clone https://github.com/YoungnohLee/etc.git # remote origin 에는 clone 해온 remote url 이 저장되어있음.

### 소스 기록
$ git add . # 모든 변경사항을 git 에 추가해줌
$ git remote show origin # remote origin 에 원격 저장소 주소가 잘 등록되었는지 확인

### 소스 커밋
$ git status # 파일의 추적 상태 ## Staged 상태의 파일은 아직 커밋된 상태가 아님.
$ git commit -m "MY COMMMIT MESSAGE"

### 소스 푸시
$ git pull origin main # main branch 를 pull 하여 로컬 저장소와 동기화
$ git push origin main


### 브랜치
$ git branch 
$ git branch BRANCH_NAME 
$ git checkout BRANCH_NAME # 새로운 브랜치로 접속

$ git push origin BRANCH_NAME
$ git branch -d OLD_BRANCH_NAME

```