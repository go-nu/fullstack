package com.example.p0718;

import androidx.lifecycle.LiveData;
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
public interface ExpenseDao {
    // 새로운 지출 데이터를 DB에 저장
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(expense: Expense)

    @Query("SELECT * FROM expense_table ORDER BY date DESC")
    fun getAllExpenses(): LiveData<List<Expense>>

    @Query("SELECT SUM(amount) FROM expense_table WHERE date = :today")
    fun getTodayTotal(today: String): LiveData<Int>
}
