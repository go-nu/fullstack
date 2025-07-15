package com.example.abc2

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class LoginActivity : AppCompatActivity() {

    private lateinit var etEmail: EditText
    private lateinit var etPassword: EditText
    private lateinit var btnLogin: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        etEmail = findViewById(R.id.etEmail)
        etPassword = findViewById(R.id.etPassword)
        btnLogin = findViewById(R.id.btnLogin)

        btnLogin.setOnClickListener {
            val email = etEmail.text.toString().trim()
            val password = etPassword.text.toString().trim()

            if(email.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "이메일과 비밀번호를 입력해주세요.", Toast.LENGTH_SHORT).show()
            } else {
                if(validateEmail(email) && validatePassword(password)) {
                    // 로그인 성공 시 동작 (여기선 예시 Toast)
                    Toast.makeText(this, "로그인 성공!", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "이메일 또는 비밀번호 형식이 올바르지 않습니다.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun validateEmail(email: String): Boolean {
        // 간단 이메일 패턴 검사
        return android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()
    }

    private fun validatePassword(password: String): Boolean {
        // 비밀번호 최소 6자 검사 예시
        return password.length >= 6
    }
}