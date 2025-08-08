// Firebase 설정
const firebaseConfig = {
    // apiKey: "your-api-key",
    // authDomain: "your-project.firebaseapp.com",
    // projectId: "your-project-id",
    // storageBucket: "your-project.appspot.com",
    // messagingSenderId: "your-sender-id",
    // appId: "your-app-id"
    apiKey: "AIzaSyA-wLOJbMyGco-GdUdIHRz6OZVRvf-Muh4",
    authDomain: "fdsfds-40ba3.firebaseapp.com",
    projectId: "fdsfds-40ba3",
    storageBucket: "fdsfds-40ba3.firebasestorage.app",
    messagingSenderId: "944730944951",
    appId: "1:944730944951:web:802e012d9bf8f084f56432"
};

// Firebase 초기화
firebase.initializeApp(firebaseConfig);

// Firestore 데이터베이스 참조
const db = firebase.firestore(); 