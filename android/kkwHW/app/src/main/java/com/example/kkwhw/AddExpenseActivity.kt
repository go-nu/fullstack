package com.example.kkwhw

import android.app.DatePickerDialog
import android.os.Bundle
import android.widget.ArrayAdapter
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.example.kkwhw.databinding.ActivityAddExpenseBinding
import com.example.kkwhw.model.Expense
import com.example.kkwhw.viewmodel.ExpenseViewModel
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

class AddExpenseActivity : AppCompatActivity() {
    private lateinit var binding: ActivityAddExpenseBinding
    private val viewModel: ExpenseViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAddExpenseBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val categories = listOf("식비", "교통비", "쇼핑", "기타")
        binding.spinnerCategory.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item, // ✅ 시스템 제공 레이아웃
            categories
        )

        val today = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())
        binding.editTextDate.setText(today)

        binding.editTextDate.setOnClickListener {
            val cal = Calendar.getInstance()
            DatePickerDialog(
                this,
                { _, y, m, d ->
                    val selectedDate = "%04d-%02d-%02d".format(y, m + 1, d)
                    binding.editTextDate.setText(selectedDate)
                },
                cal.get(Calendar.YEAR),
                cal.get(Calendar.MONTH),
                cal.get(Calendar.DAY_OF_MONTH)
            ).show()
        }

        binding.buttonSave.setOnClickListener {
            val name = binding.editTextName.text.toString()
            val amount = binding.editTextAmount.text.toString().toIntOrNull() ?: 0
            val date = binding.editTextDate.text.toString()
            val category = binding.spinnerCategory.selectedItem.toString()

            if (name.isNotBlank() && amount > 0) {
                val expense = Expense(name = name, amount = amount, date = date, category = category)
                viewModel.insert(expense)
                finish()
            }
        }
    }
}
