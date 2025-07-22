package com.example.app0722
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.viewModelScope
import com.example.app0722.EmotionDatabase
import com.example.app0722.Emotion
import kotlinx.coroutines.launch

// 데이터 관리: Emotion 객체를 저장, 불러오기, 삭제 하는 등의 데이터베이스 작업
class EmotionViewModel(application: Application) : AndroidViewModel(application) {
    private val emotionDao = EmotionDatabase.getDatabase(application).emotionDao()
    val emotions: LiveData<List<Emotion>> = emotionDao.getAll()

    // 데이터 추가
    fun addEmotion(emotion: Emotion) = viewModelScope.launch {
        emotionDao.insert(emotion)
    }

    // 데이터 삭제
    fun deleteEmotion(emotion: Emotion) = viewModelScope.launch {
        emotionDao.delete(emotion)
    }
}