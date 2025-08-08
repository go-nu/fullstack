# AI 강의 사이트

AI 강의 신청 사이트로, Firebase Firestore를 사용하여 신청 데이터를 관리하는 웹 애플리케이션입니다.

## 🚀 주요 기능

- **수강신청 폼**: 이름, 이메일, 휴대폰, 신청동기 입력
- **Firebase 연동**: 실시간 데이터 저장 및 관리
- **관리자 대시보드**: 신청 데이터 조회 및 삭제
- **반응형 디자인**: 모바일, 태블릿, 데스크톱 지원
- **실시간 업데이트**: 신청 데이터 실시간 반영

## 📋 설정 방법

### 1. Firebase 프로젝트 생성

1. [Firebase Console](https://console.firebase.google.com/)에 접속
2. 새 프로젝트 생성
3. Firestore Database 활성화
4. 보안 규칙 설정 (테스트 모드로 시작)

### 2. Firebase 설정 정보 입력

`firebase-config.js` 파일에서 Firebase 프로젝트 설정 정보를 입력하세요:

```javascript
const firebaseConfig = {
    apiKey: "your-api-key",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project.appspot.com",
    messagingSenderId: "your-sender-id",
    appId: "your-app-id"
};
```

### 3. Firestore 보안 규칙 설정

Firebase Console에서 Firestore > 규칙 탭에서 다음 규칙을 설정하세요:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /applications/{document} {
      allow read, write: if true;  // 테스트용 (실제 운영시에는 인증 추가 필요)
    }
  }
}
```

## 📁 파일 구조

```
├── index.html              # 메인 HTML 파일
├── firebase-config.js      # Firebase 설정
├── app.js                  # 메인 JavaScript 로직
└── README.md              # 프로젝트 설명서
```

## 🎯 사용법

### 일반 사용자
1. 웹사이트 접속
2. "신청하기" 버튼 클릭
3. 수강신청 폼 작성 및 제출

### 관리자
1. 우상단 "관리자" 버튼 클릭
2. 수강신청 대시보드에서 신청 데이터 확인
3. 필요시 개별 신청 삭제 가능

## 🔧 기술 스택

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Database**: Firebase Firestore
- **Styling**: CSS Grid, Flexbox, CSS Animations
- **Deployment**: 정적 웹 호스팅 (GitHub Pages, Netlify 등)

## 🚀 배포 방법

### GitHub Pages
1. GitHub 저장소 생성
2. 파일 업로드
3. Settings > Pages에서 배포 설정

### Netlify
1. Netlify 계정 생성
2. GitHub 저장소 연결
3. 자동 배포 설정

## 📱 반응형 지원

- **모바일**: 320px 이상
- **태블릿**: 768px 이상  
- **데스크톱**: 1024px 이상

## 🔒 보안 고려사항

현재 버전은 테스트용으로 모든 읽기/쓰기 권한을 허용합니다. 실제 운영 환경에서는:

1. Firebase Authentication 추가
2. 관리자 인증 시스템 구현
3. 적절한 보안 규칙 설정
4. HTTPS 강제 적용

## 🐛 문제 해결

### Firebase 연결 오류
- Firebase 설정 정보 확인
- 인터넷 연결 상태 확인
- 브라우저 콘솔에서 오류 메시지 확인

### 데이터가 표시되지 않음
- Firestore 데이터베이스 활성화 확인
- 보안 규칙 설정 확인
- 브라우저 캐시 삭제

## 📞 지원

문제가 발생하거나 개선 사항이 있으시면 이슈를 등록해주세요.

---

**개발자**: AI Academy Team  
**버전**: 1.0.0  
**최종 업데이트**: 2024년 