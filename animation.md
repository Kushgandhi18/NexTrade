. Animation Philosophy (Don’t overdo it)
Fast, subtle, purposeful
Never block trading actions
Data changes → smooth transitions (not jumps)

👉 Think: “alive, not flashy”

⚙️ 2. Core Animation System
⏱ Timing
Micro interactions: 150–250ms
Page transitions: 300–400ms
Charts: 500–800ms
🧠 Easing
Use: ease-out (default)
Premium feel: cubic-bezier(0.22, 1, 0.36, 1)
🧩 3. Key Animations (Screen-by-Screen)
🏠 Dashboard Animations
✨ 1. Card Entry (on load)
Fade in + slight upward motion

Figma:

Opacity: 0 → 100
Y: 20 → 0

React (Framer Motion):

<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.3 }}
/>
📊 2. Portfolio Value Count-Up
Numbers animate instead of jumping

Example:

$12,450 → smoothly increments

Use:

react-countup
📈 3. Watchlist Hover Effect
Card lifts slightly
whileHover={{ scale: 1.03, y: -4 }}
📊 Stock Detail Page
📉 4. Chart Draw Animation (CRITICAL)
Line grows from left → right
AI prediction line appears after

Figma:

Smart animate with mask

React:

Chart.js animation or SVG path animation
🤖 5. AI Prediction Reveal

Flow:

Loading shimmer
Fade in prediction
Confidence bar fills
initial={{ width: 0 }}
animate={{ width: "87%" }}
🔵 6. BUY / SELL Button Feedback

On click:

Slight press effect
Color flash
whileTap={{ scale: 0.95 }}
🤖 AI Prediction Page
🧠 7. Model Switching Animation

When switching:

LSTM → GRU → Transformer

Animation:

Slide + fade
initial={{ x: 20, opacity: 0 }}
animate={{ x: 0, opacity: 1 }}
🎯 8. Confidence Gauge Animation
Circular progress fills smoothly

Figma:

Arc rotation

React:

SVG stroke animation
💰 Trade Execution
⚡ 9. Order Placement

Flow:

Button press
Loading spinner
Success checkmark
✅ Success Animation
Scale up + bounce
initial={{ scale: 0 }}
animate={{ scale: 1 }}
transition={{ type: "spring", stiffness: 200 }}
📁 Portfolio Page
📊 10. Pie Chart Animation
Segments grow from 0
📈 11. Profit/Loss Change
Number flashes green/red briefly
🌍 Navigation Animations
🔁 12. Page Transitions
Slide + fade
initial={{ opacity: 0, x: 30 }}
animate={{ opacity: 1, x: 0 }}
exit={{ opacity: 0, x: -30 }}
📱 13. Bottom Nav Indicator
Smooth sliding underline
✨ 4. Premium Micro-Interactions (GAME CHANGER)
🔥 AI Signal Pulse

When “BUY” appears:

Soft glowing pulse
box-shadow: 0 0 0 rgba(0, 208, 156, 0.7);
animation: pulse 1.5s infinite;
💡 Tooltip Fade
Appears instantly but fades out softly
📡 Live Price Update
Price changes animate:
Flash green/red
Slight bounce
🧲 Magnetic Buttons
Cursor pulls slightly toward button (advanced but 🔥)
🎨 5. Figma Animation Setup
Use:
Smart Animate
After Delay triggers
Overlay (for modals)
Example Flow:
Dashboard → (On click stock)
Smart Animate → Stock Detail

Buy Button → Open Overlay → Order Modal

Order Success → Auto Animate → Success Screen
⚠️ 6. Common Mistakes (Avoid These)

❌ Slow animations (feels laggy)
❌ Too many effects (looks like a game)
❌ Blocking UI during trading
❌ Random motion without purpose

🚀 7. What Makes It “Premium”

✔ Smooth chart transitions
✔ Instant feedback on trades
✔ AI animations feel “alive”
✔ Micro-interactions everywhere