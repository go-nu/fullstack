package com.example.myapplication0716
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication0716.databinding.ItemTodoBinding

class TodoAdapter(
    private val todoList: MutableList<Todo>
) : RecyclerView.Adapter<TodoAdapter.TodoViewHolder>() {

    inner class TodoViewHolder(val binding: ItemTodoBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TodoViewHolder {
        val binding = ItemTodoBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return TodoViewHolder(binding)
    }

    override fun onBindViewHolder(holder: TodoViewHolder, position: Int) {
        val todo = todoList[position]

        holder.binding.tvTask.text = todo.task
        holder.binding.checkBox.isChecked = todo.isChecked

        holder.binding.checkBox.setOnCheckedChangeListener { _, isChecked ->
            todo.isChecked = isChecked
        }
    }

    override fun getItemCount(): Int = todoList.size

    fun addTodo(todo: Todo) {
        todoList.add(todo)
        notifyItemInserted(todoList.size - 1)
    }

    fun removeTodo(position: Int) {
        todoList.removeAt(position)
        notifyItemRemoved(position)
    }
}
