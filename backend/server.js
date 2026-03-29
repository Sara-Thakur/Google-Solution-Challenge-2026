const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json());

// 1. MongoDB Connection (Fixed for latest Mongoose)
mongoose.connect('mongodb://127.0.0.1:27017/bankProject')
    .then(() => console.log("✅ MongoDB Connected Successfully!"))
    .catch(err => console.log("❌ Connection Error:", err));

// 2. User Schema (For Login/Signup)
const UserSchema = new mongoose.Schema({
    name: String,
    email: { type: String, unique: true, required: true },
    password: { type: String, required: true }
});
const User = mongoose.model('User', UserSchema);

// 3. Audit Schema (For History)
const AuditSchema = new mongoose.Schema({
    userName: String,
    loanAmount: String,
    status: String,
    date: { type: Date, default: Date.now }
});
const Audit = mongoose.model('Audit', AuditSchema);

// --- ROUTES ---

// Signup API
app.post('/api/signup', async (req, res) => {
    try {
        const newUser = new User(req.body);
        await newUser.save();
        res.status(201).json({ message: "Account Created!", name: newUser.name });
    } catch (err) {
        res.status(400).json({ error: "Email already exists!" });
    }
});

// Login API
app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;
    const user = await User.findOne({ email, password });
    if (user) {
        res.json({ message: "Login Successful", name: user.name });
    } else {
        res.status(401).json({ error: "Invalid Email or Password" });
    }
});

// Save Audit API
app.post('/api/save-audit', async (req, res) => {
    try {
        const newAudit = new Audit(req.body);
        await newAudit.save();
        res.status(201).json({ message: "Audit Saved!" });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Get History API
app.get('/api/history', async (req, res) => {
    try {
        const history = await Audit.find().sort({ date: -1 });
        res.json(history);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.listen(3000, () => console.log("🚀 Server running on http://localhost:3000"));