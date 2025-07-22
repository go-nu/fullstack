package com.example.kkwhw

import androidx.lifecycle.LiveData
import androidx.room.*
import com.example.kkwhw.model.Expense
import com.example.kkwhw.model.MonthlyTotal

@Dao
interface ExpenseDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(expense: Expense)

    @Delete
    suspend fun delete(expense: Expense)

    @Query("SELECT * FROM expenses ORDER BY date DESC")
    fun getAllExpenses(): LiveData<List<Expense>>

    @Query("SELECT * FROM expenses WHERE category = :category")
    fun getExpensesByCategory(category: String): LiveData<List<Expense>>

    @Query("SELECT SUM(amount) FROM expenses WHERE date = :date")
    fun getTotalAmountByDate(date: String): LiveData<Int>

    @Query("SELECT date, SUM(amount) as total FROM expenses GROUP BY date")
    fun getMonthlyTotals(): LiveData<List<MonthlyTotal>>
}
