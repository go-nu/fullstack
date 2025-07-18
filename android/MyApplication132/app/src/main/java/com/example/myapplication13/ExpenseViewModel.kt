package com.example.myapplication13

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
// ViewModel 클래스
// UI(Activity)와 데이터(Room DB) 사이를 연결하는 중간 관리자
// LiveData로 데이터를 관찰하고 UI 업데이트를 도와줌
// insert, getAllExpense 등의 메서드를 가짐