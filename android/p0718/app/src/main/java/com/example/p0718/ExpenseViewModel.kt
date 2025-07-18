package com.example.p0718

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class ExpenseViewModel(application: Application) : AndroidViewModel(application) {

    private val expenseDao = ExpenseDatabase.getDatabase(application).expenseDao()
    val allExpenses: LiveData<List<Expense>> = expenseDao.getAllExpenses()

    fun insertExpense(expense: Expense) {
        viewModelScope.launch(Dispatchers.IO) {
            expenseDao.insert(expense)
        }
    }

    fun getTodayTotal(today: String): LiveData<Int> {
        return expenseDao.getTodayTotal(today)
    }
}