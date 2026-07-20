package com.altrodav.friendcircle.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

// 1. Data Class representing a Friend
data class Friend(
    val id: String, // Stable key
    val name: String,
    val avatarUrl: String,
    val isOnline: Boolean = false
)

// 2. Mocking the ProfileHeader pattern as requested
@Composable
fun ProfileHeader(name: String, avatarUrl: String, isOnline: Boolean) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
    ) {
        // Mock Avatar (In a real app, use AsyncImage from Coil)
        Box(
            modifier = Modifier
                .size(48.dp)
                .padding(4.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(text = name.first().toString(), style = MaterialTheme.typography.headlineSmall)
        }
        
        Spacer(modifier = Modifier.width(12.dp))
        
        Column {
            Text(text = name, style = MaterialTheme.typography.bodyLarge)
            Text(
                text = if (isOnline) "Online" else "Offline",
                style = MaterialTheme.typography.bodySmall,
                color = if (isOnline) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// 3. The FriendRow component
@Composable
fun FriendRow(friend: Friend) {
    // Reusing the ProfileHeader pattern
    ProfileHeader(
        name = friend.name,
        avatarUrl = friend.avatarUrl,
        isOnline = friend.isOnline
    )
}

// 4. The main FriendList Composable with LazyColumn
@Composable
fun FriendList(friends: List<Friend>) {
    // Handle the empty state
    if (friends.isEmpty()) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "It's a little quiet here.\nAdd some friends to get started!",
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    } else {
        // Efficiently display scrollable list
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp), // Content padding
            verticalArrangement = Arrangement.spacedBy(8.dp) // Spacing between items
        ) {
            items(
                items = friends,
                key = { friend -> friend.id } // Stable key for performance and correctness
            ) { friend ->
                FriendRow(friend = friend)
            }
        }
    }
}

// 5. Sample Data for testing (20+ friends)
val sampleFriends = List(25) { index ->
    Friend(
        id = "friend_$index",
        name = "Friend ${index + 1}",
        avatarUrl = "https://example.com/avatar/$index.png",
        isOnline = index % 3 == 0 // Every 3rd friend is online
    )
}
