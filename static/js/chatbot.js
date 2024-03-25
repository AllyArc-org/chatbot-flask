$(document).ready(function () {
  var chatbotMessages = $("#chatbot-messages");

  var userInput = $("#user-input");

  var sendBtn = $("#send-btn");

  var suggestionButtons = $(".suggestion-btn");

  var loading = $("#loading");

  var darkModeBtn = $("#dark-mode-btn");

  var newChatBtn = $("#new-chat-btn");

  // Send user message to chatbot and display response
// Send user message to chatbot and display response
function sendMessage(message) {
  // Display user message
  var userMessage = $('<div class="message user-message"></div>');
  var userMessageBubble = $('<div class="message-bubble"></div>').text(message);
  userMessage.append(userMessageBubble);
  chatbotMessages.append(userMessage);
  chatbotMessages.animate({ scrollTop: chatbotMessages[0].scrollHeight }, 500);

  // Show loading animation
  var loadingMessage = $('<div class="message chatbot-message"></div>');
  var loadingMessageBubble = $('<div class="message-bubble"></div>');
  var loadingContent = $('<div class="loading"></div>');
  loadingContent.append('<span class="dot"></span><span class="dot"></span><span class="dot"></span>');
  loadingMessageBubble.append(loadingContent);
  loadingMessage.append(loadingMessageBubble);
  chatbotMessages.append(loadingMessage);
  chatbotMessages.animate({ scrollTop: chatbotMessages[0].scrollHeight }, 500);

  // Send message to server and display chatbot response
  $.ajax({
    url: "/chatbot",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({ input: message }),
    success: function (response) {
      loadingMessage.remove();
      var assistantResponse = response.response;
      var chatbotMessage = $('<div class="message chatbot-message"></div>');
      var chatbotAvatar = $('<div class="chatbot-avatar"></div>').text("C");
      var chatbotMessageBubble = $('<div class="message-bubble"></div>').text(assistantResponse);
      chatbotMessage.append(chatbotAvatar);
      chatbotMessage.append(chatbotMessageBubble);
      chatbotMessages.append(chatbotMessage);
      chatbotMessages.animate({ scrollTop: chatbotMessages[0].scrollHeight }, 500);
    },
  });
}

  // Handle send button click

  sendBtn.click(function () {
    var message = userInput.val();

    if (message.trim() !== "") {
      sendMessage(message);

      userInput.val("");
    }
  });

  // Handle user input submission

  userInput.keypress(function (event) {
    if (event.which === 13) {
      sendBtn.click();
    }
  });

  // Handle suggestion button clicks

  suggestionButtons.click(function () {
    var message = $(this).data("value");

    sendMessage(message);
  });

  // Toggle dark mode

  darkModeBtn.click(function () {
    $("body").toggleClass("dark-mode");
  });

  // Start new chat

  newChatBtn.click(function () {
    chatbotMessages.empty();
  });
});
