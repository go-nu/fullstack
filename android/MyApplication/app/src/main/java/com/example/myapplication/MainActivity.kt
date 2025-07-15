package com.example.myapplication

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView

class MainActivity : AppCompatActivity() {

    var num1 = ""
    var num2 = ""
    var operator = ""
    lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultTextView = findViewById(R.id.resultTextView)

        // 숫자 버튼
        val button0 = findViewById<Button>(R.id.button0)
        val button1 = findViewById<Button>(R.id.button1)
        val button2 = findViewById<Button>(R.id.button2)
        val button3 = findViewById<Button>(R.id.button3)
        val button4 = findViewById<Button>(R.id.button4)
        val button5 = findViewById<Button>(R.id.button5)
        val button6 = findViewById<Button>(R.id.button6)
        val button7 = findViewById<Button>(R.id.button7)
        val button8 = findViewById<Button>(R.id.button8)
        val button9 = findViewById<Button>(R.id.button9)

        val buttons = listOf(button0, button1, button2, button3, button4, button5, button6, button7, button8, button9)

        buttons.forEachIndexed { index, button ->
            button.setOnClickListener { numberClicked(index.toString()) }
        }

        // 연산자 버튼
        findViewById<Button>(R.id.buttonPlus).setOnClickListener { operatorClicked("+") }
        findViewById<Button>(R.id.buttonMinus).setOnClickListener { operatorClicked("-") }
        findViewById<Button>(R.id.buttonMultiply).setOnClickListener { operatorClicked("×") }
        findViewById<Button>(R.id.buttonDivide).setOnClickListener { operatorClicked("÷") }

        // = 버튼
        findViewById<Button>(R.id.buttonEqual).setOnClickListener { calculate() }

        // C 버튼
        findViewById<Button>(R.id.buttonClear).setOnClickListener { clear() }
    }

    private fun numberClicked(number: String) {
        if (operator.isEmpty()) {
            num1 += number
            resultTextView.text = num1
        } else {
            num2 += number
            resultTextView.text = num2
        }
    }

    private fun operatorClicked(op: String) {
        if (num1.isNotEmpty()) {
            operator = op
        }
    }

    private fun calculate() {
        if (num1.isNotEmpty() && num2.isNotEmpty()) {
            val result = when (operator) {
                "+" -> num1.toInt() + num2.toInt()
                "-" -> num1.toInt() - num2.toInt()
                "×" -> num1.toInt() * num2.toInt()
                "÷" -> if (num2 != "0") num1.toInt() / num2.toInt() else "오류"
                else -> "오류"
            }
            resultTextView.text = "결과: $result"
            clearAfterCalc(result.toString())
        }
    }

    private fun clear() {
        num1 = ""
        num2 = ""
        operator = ""
        resultTextView.text = "0"
    }

    private fun clearAfterCalc(result: String) {
        num1 = result
        num2 = ""
        operator = ""
    }
}