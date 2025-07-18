package com.example.p0718

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import androidx.lifecycle.ViewModelProvider
import java.time.LocalDate

class AddExpenseActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("AddExpenseActivity", "onCreate 호출됨")
        setContentView(R.layout.activity_add_expense)

        val viewModel = ViewModelProvider(this)[ExpenseViewModel::class.java]

        findViewById<Button>(R.id.btn_save).setOnClickListener {
            val title = findViewById<EditText>(R.id.et_title).text.toString()
            val amount = findViewById<EditText>(R.id.et_amount).text.toString().toInt()
            val category = findViewById<EditText>(R.id.et_category).text.toString()
            val date = LocalDate.now().toString()

            val expense = Expense(title = title, amount = amount, category = category, date = date)
            Log.d("AddExpenseActivity", "저장할 지출: $expense")
            viewModel.insertExpense(expense)
            finish()
        }
    }
}
