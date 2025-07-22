package com.example.app0722

import android.os.Bundle
import android.widget.ArrayAdapter
import androidx.activity.ComponentActivity
import androidx.activity.viewModels
import com.example.app0722.databinding.ActivityAddEmotionBinding
import java.text.SimpleDateFormat
import java.util.*

// AddEmotionActivity 사용자가 자신의 감정과 관련된 내용을 입력 및 저장
class AddEmotionActivity : ComponentActivity() {
    private lateinit var binding: ActivityAddEmotionBinding
    private val viewModel: EmotionViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAddEmotionBinding.inflate(layoutInflater)
        setContentView(binding.root)
        // 감정 선택 : 미리 정의된 감정 중 하나 선택할 수 있는 dropdown (spinner) 제공
        val feelings = listOf("🙂 행복", "😢 슬픔", "😡 화남", "😨 불안", "😐 무표정")
        val spinnerAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, feelings)
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerFeeling.adapter = spinnerAdapter

        binding.btnSave.setOnClickListener {
            val feeling = binding.spinnerFeeling.selectedItem.toString()
            val content = binding.etContent.text.toString() // 내용 작성
            val date = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date()) // 날짜 자동 기록

            if (content.isNotBlank()) {
                val emotion = Emotion(date = date, feeling = feeling, content = content)
                // 날짜, 감정, 내용을 담는 Emotion 객체를 생성
                viewModel.addEmotion(emotion) // 객체를 EmotionView Model에 추가
                finish()
            }
        }
    }
}
