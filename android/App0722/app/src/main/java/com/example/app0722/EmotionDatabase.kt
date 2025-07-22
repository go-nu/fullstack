package com.example.app0722
import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

// Room database 선언
@Database(entities = [Emotion::class], version = 1)
abstract class EmotionDatabase : RoomDatabase() {
    abstract fun emotionDao(): EmotionDao
    //데이터베이스에 접근하여 Emotion 데이터를 삽입, 조회, 삭제 할 수 있는 emotionDao() 메서드 제공
    companion object {
        @Volatile private var INSTANCE: EmotionDatabase? = null

        fun getDatabase(context: Context): EmotionDatabase {
            return INSTANCE ?: synchronized(this) {
                Room.databaseBuilder(
                    context.applicationContext,
                    EmotionDatabase::class.java,
                    "emotion_db"
                ).build().also { INSTANCE = it }
            }
        }
    }
}