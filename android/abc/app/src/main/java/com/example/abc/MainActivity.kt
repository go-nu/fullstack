package com.example.abc

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.TextView
import android.widget.Button
import com.google.android.material.floatingactionbutton.FloatingActionButton
import android.widget.Toast

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val helloText = findViewById<TextView>(R.id.helloText)
        val myButton = findViewById<Button>(R.id.myButton)
        val fabAdd = findViewById<FloatingActionButton>(R.id.fabAdd)

        myButton.setOnClickListener {
            helloText.text = "버튼 클릭했어요!"
        }

        fabAdd.setOnClickListener {
            Toast.makeText(this, "FAB 버튼 클릭됨!", Toast.LENGTH_SHORT).show()
        }
    }
}
