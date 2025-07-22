package com.example.app0722

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "emotion_table")
data class Emotion(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val date: String,
    val feeling: String,
    val content: String
)