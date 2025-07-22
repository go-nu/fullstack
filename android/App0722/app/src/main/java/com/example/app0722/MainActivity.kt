package com.example.app0722

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.viewModels
import androidx.recyclerview.widget.LinearLayoutManager

import com.example.app0722.databinding.ActivityMainBinding

class MainActivity : ComponentActivity() {
    private lateinit var binding: ActivityMainBinding
    // 감정 데이터 관리, 데이터가 사라지지 않게 유지
    private val viewModel: EmotionViewModel by viewModels()

    // 초기 설정 수행
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater) // 객체 초기화
        setContentView(binding.root) // activity_main.xml 레이아웃으로 설정

        val adapter = EmotionAdapter { emotion -> viewModel.deleteEmotion(emotion) }
        binding.rvEmotions.layoutManager = LinearLayoutManager(this)
        binding.rvEmotions.adapter = adapter

        viewModel.emotions.observe(this) {
            adapter.submitList(it)
        }

        binding.btnAdd.setOnClickListener {
            startActivity(Intent(this, AddEmotionActivity::class.java))
        }
    }
}