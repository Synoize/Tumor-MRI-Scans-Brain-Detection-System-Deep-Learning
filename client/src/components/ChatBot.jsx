import React, { useState, useEffect, useRef } from "react";


const ChatBot = () => {
    ;
    const [messages, setMessages] = useState([
        { role: "bot", content: "Hi — I’m your assistant. Ask me anything!" },
    ]);
    const [input, setInput] = useState("");
    const chatRef = useRef(null);

    useEffect(() => {
        if (chatRef.current) {
            chatRef.current.scrollTop = chatRef.current.scrollHeight;
        }
    }, [messages]);

    // Save chat to localStorage
    useEffect(() => {
        localStorage.setItem("chatHistory", JSON.stringify(messages));
    }, [messages]);

    useEffect(() => {
        const saved = JSON.parse(localStorage.getItem("chatHistory")) || [];
        if (saved.length > 0) setMessages(saved);
    }, []);

    const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

    console.log(GEMINI_API_KEY);


    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage = { role: "user", content: input.trim() };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");

        const loadingMsg = { role: "bot", content: "Thinking..." };
        setMessages((prev) => [...prev, loadingMsg]);

        const tumorKeywords = [
            "tumor", "brain tumor", "cancer", "tumour", "malignant", "benign",
            "radiation", "chemotherapy", "mri", "treatment", "symptom", "scan"
        ];
        const isTumorQuery = tumorKeywords.some((word) =>
            userMessage.content.toLowerCase().includes(word)
        );

        const prompt = isTumorQuery
            ? `You are a medical AI assistant specialized in tumors. 
         Only answer questions related to tumors, their types, harms, diagnosis, and treatment.
         If user asks unrelated things, say: "I can only answer tumor-related queries." 
         Question: ${userMessage.content}`
            : userMessage.content;

        try {
            const res = await fetch(
                `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        contents: [{ parts: [{ text: prompt }] }],
                    }),
                }
            );

            const data = await res.json();

            const reply =
                data?.candidates?.[0]?.content?.parts?.[0]?.text ||
                "Sorry, I didn’t understand that.";

            setMessages((prev) => [
                ...prev.slice(0, -1), // remove "Thinking..."
                { role: "bot", content: reply },
            ]);
        } catch (error) {
            setMessages((prev) => [
                ...prev.slice(0, -1),
                { role: "bot", content: `Error: ${error.message}` },
            ]);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter") sendMessage();
    };

    const clearChat = () => {
        setMessages([{ role: "bot", content: "Hi — I’m your assistant. Ask me anything!" }]);
        localStorage.removeItem("chatHistory");
    };

    return (
        <div className="flex flex-col h-[500px] bg-white/20 backdrop-blur-xl">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-special-red text-white">
                <h1 className="text-lg font-semibold">Rahi - ChatBot</h1>
                <button
                    onClick={clearChat}
                    className="text-sm bg-white text-special-red px-3 py-1 rounded-lg hover:bg-gray-50"
                >
                    Clear Chat
                </button>
            </div>

            {/* Chat Messages */}
            <div
                ref={chatRef}
                className="flex-1 overflow-y-auto p-4 space-y-3"
            >
                {messages.map((msg, i) => (
                    <div
                        key={i}
                        className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"
                            }`}
                    >
                        <div
                            className={`px-4 py-2 rounded-2xl max-w-[75%] text-sm shadow-md whitespace-pre-line ${msg.role === "user"
                                    ? "bg-special-red text-white"
                                    : "bg-white/80 backdrop-blur-xl text-gray-900"
                                }`}
                        >
                            {msg.content}
                        </div>
                    </div>
                ))}
            </div>

            {/* Input Field */}
            <div className="p-4 flex items-center gap-3 bg-white/80 backdrop-blur-xl">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your message..."
                    className="flex-1 border border-gray-400 rounded-xl px-4 py-2 outline-none focus:border-special-red"
                />
                <button
                    onClick={sendMessage}
                    className=" cursor-pointer bg-special-red/90 text-white px-4 py-2 rounded-xl hover:bg-special-red"
                >
                    Send
                </button>
            </div>
        </div>
    );
};

export default ChatBot;
