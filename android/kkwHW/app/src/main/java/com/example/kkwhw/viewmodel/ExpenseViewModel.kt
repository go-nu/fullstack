package com.example.kkwhw.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.viewModelScope
import com.example.kkwhw.ExpenseDatabase
import com.example.kkwhw.model.Expense
import kotlinx.coroutines.launch

class ExpenseViewModel(application: Application) : AndroidViewModel(application) {
    private val dao = ExpenseDatabase.Companion.getDatabase(application).expenseDao()

    val allExpenses: LiveData<List<Expense>> = dao.getAllExpenses()
    fun getTotalAmountByDate(date: String) = dao.getTotalAmountByDate(date)
    fun getExpensesByCategory(category: String) = dao.getExpensesByCategory(category)

    fun insert(expense: Expense) = viewModelScope.launch {
        dao.insert(expense)
    }

    fun delete(expense: Expense) = viewModelScope.launch {
        dao.delete(expense)
    }
}