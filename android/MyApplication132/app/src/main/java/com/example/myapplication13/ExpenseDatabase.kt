package com.example.myapplication13

import android.content.Context
import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.Room

@Database(entities = [Expense::class], version = 1)
// RoomDatabase를 상속하여 데이터베이스 클래스를 정의, 이 클래스를 통해 DAO 객체를 제공 받음
abstract class ExpenseDatabase : RoomDatabase() {
    abstract fun expenseDao(): ExpenseDao

    companion object {
        @Volatile private var INSTANCE: ExpenseDatabase? = null

        fun getDatabase(context: Context): ExpenseDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    ExpenseDatabase::class.java,
                    "expense_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
// Room DB를 상속하는 추상 클래스
// 실제 Room DB 인스턴스를 생성하고 Dao 접근 메서드를 포함
// Singleton