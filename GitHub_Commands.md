# 🚀 GitHub 업로드 명령어 가이드

## 📋 1단계: GitHub에서 새 저장소 생성
1. https://github.com/Jirehhyeon/ 접속
2. **"New repository"** 클릭
3. 저장소 이름: `SysWatch-Pro-Quantum-Nexus-Infinity`
4. 설명: `🌌 Next-Generation Quantum Computing Holographic System Monitor`
5. **Public** 선택
6. ✅ **"Add a README file"** 체크
7. **"Create repository"** 클릭

## 🛠️ 2단계: 로컬에서 Git 명령어 실행

### Git 설치 확인:
```bash
git --version
```

### 프로젝트 폴더로 이동:
```bash
cd C:\Users\WIN10\Desktop\SysWatch-Pro
```

### Git 저장소 초기화:
```bash
git init
```

### GitHub 원격 저장소 연결:
```bash
git remote add origin https://github.com/Jirehhyeon/SysWatch-Pro-Quantum-Nexus-Infinity.git
```

### 기본 브랜치 설정:
```bash
git branch -M main
```

### 모든 파일 추가:
```bash
git add .
```

### 첫 커밋 생성:
```bash
git commit -m "🌌 Initial commit: SysWatch Pro QUANTUM NEXUS INFINITY

✨ Features:
- 165fps Holographic 3D Interface
- Deep Hardware Monitoring (CPU Registers, Memory Sectors)  
- AI Prediction Engine with Neural Networks
- Quantum Security Scanner
- Real-time Particle Fields & Crystal Matrix
- Floating Quantum Panels
- Multi-threading Performance Optimization

🚀 Ready for quantum computing era system monitoring!"
```

### GitHub에 업로드:
```bash
git push -u origin main
```

## 🎯 3단계: 완료 확인
https://github.com/Jirehhyeon/SysWatch-Pro-Quantum-Nexus-Infinity 에서 확인

---

## 🔄 업데이트할 때 (나중에):

### 변경사항 확인:
```bash
git status
```

### 변경된 파일 추가:
```bash
git add .
```

### 커밋:
```bash
git commit -m "📝 Update: 설명"
```

### 업로드:
```bash
git push
```

---

## ⚠️ 문제 해결

### Git 설치 안 됨:
- https://git-scm.com/download/win 에서 설치

### 권한 오류:
```bash
git config --global user.name "Jirehhyeon"
git config --global user.email "your-email@example.com"
```

### 저장소가 이미 존재:
```bash
git remote -v
git remote remove origin
git remote add origin https://github.com/Jirehhyeon/SysWatch-Pro-Quantum-Nexus-Infinity.git
```

### 첫 push 실패:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## 🎉 완료!
✅ 프로젝트가 GitHub에 성공적으로 업로드됩니다!
✅ 다른 사람들이 다운로드하고 사용할 수 있습니다!
✅ 포트폴리오로 활용할 수 있습니다!