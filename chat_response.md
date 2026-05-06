Below is a **complete, readyâtoâcopy** HTML file that creates a clean, responsive, and accessible form page.  
It uses only plain HTMLâ¯+â¯CSS (no external libraries) but you can easily swap the CSS for a framework like Tailwind or Bootstrap if you prefer.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Form â My Awesome Site</title>

    <!-- -------------------------------------------------
         Basic styling â feel free to replace with your own
         ------------------------------------------------- -->
    <style>
        /* Reset a few defaults for consistency */
        *, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.5;
            background: #f4f7fb;
            color: #333;
            padding: 2rem;
        }

        h1 {
            text-align: center;
            margin-bottom: 1.5rem;
            color: #2c3e50;
        }

        .form-wrapper {
            max-width: 560px;
            margin: 0 auto;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,.08);
            padding: 2rem;
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            margin-bottom: .4rem;
            font-weight: 600;
        }

        input,
        textarea,
        select {
            width: 100%;
            padding: .75rem 1rem;
            border: 1px solid #cbd5e0;
            border-radius: 4px;
            font-size: 1rem;
            transition: border-color .2s, box-shadow .2s;
        }

        input:focus,
        textarea:focus,
        select:focus {
            outline: none;
            border-color: #5a67d8;
            box-shadow: 0 0 0 3px rgba(90,103,216,.2);
        }

        textarea {
            min-height: 120px;
            resize: vertical;
        }

        .error-msg {
            color: #e53e3e;
            font-size: .875rem;
            margin-top: .3rem;
            display: none;
        }

        .btn-primary {
            display: inline-block;
            width: 100%;
            padding: .85rem;
            background: #5a67d8;
            color: #fff;
            border: none;
            border-radius: 4px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background .2s;
        }

        .btn-primary:hover { background: #434190; }

        .success-msg {
            background: #e6fffa;
            border: 1px solid #b2f5ea;
            color: #2c7a7b;
            padding: .75rem 1rem;
            border-radius: 4px;
            margin-top: 1rem;
            display: none;
        }

        /* ---- Responsive ---- */
        @media (max-width: 480px) {
            body { padding: 1rem; }
            .form-wrapper { padding: 1.5rem; }
        }
    </style>
</head>
<body>

    <h1>Contact Us</h1>

    <section class="form-wrapper" aria-labelledby="contact-form-heading">
        <h2 id="contact-form-heading" class="sr-only">Contact form</h2>

        <form id="contactForm" novalidate>
            <!-- Name -->
            <div class="form-group">
                <label for="name">Full Name <span aria-hidden="true">*</span></label>
                <input type="text" id="name" name="name" required aria-required="true" autocomplete="name">
                <p class="error-msg" id="err-name">Please enter your name.</p>
            </div>

            <!-- Email -->
            <div class="form-group">
                <label for="email">Email address <span aria-hidden="true">*</span></label>
                <input type="email" id="email" name="email" required aria-required="true" autocomplete="email">
                <p class="error-msg" id="err-email">Enter a valid email address.</p>
            </div>

            <!-- Phone (optional) -->
            <div class="form-group">
                <label for="phone">Phone number</label>
                <input type="tel" id="phone" name="phone" autocomplete="tel"
                       pattern="\\+?[0-9\\s\\-]{7,15}"
                       placeholder="+1 555 123 4567">
                <p class="error-msg" id="err-phone">Enter a valid phone number.</p>
            </div>

            <!-- Reason (select) -->
            <div class="form-group">
                <label for="reason">Reason for contact</label>
                <select id="reason" name="reason">
                    <option value="">-- Please choose --</option>
                    <option value="question">General Question</option>
                    <option value="support">Support Request</option>
                    <option value="feedback">Feedback / Suggestion</option>
                </select>
            </div>

            <!-- Message -->
            <div class="form-group">
                <label for="message">Message <span aria-hidden="true">*</span></label>
                <textarea id="message" name="message" required aria-required="true"></textarea>
                <p class="error-msg" id="err-message">Please write a message.</p>
            </div>

            <button type="submit" class="btn-primary">Send Message</button>

            <div class="success-msg" id="successMsg" role="status" aria-live="polite">
                ð Your message has been sent! Weâll get back to you shortly.
            </div>
        </form>
    </section>

    <!-- -------------------------------------------------
         Simple clientâside validation & demo âsubmissionâ
         ------------------------------------------------- -->
    <script>
        const form = document.getElementById('contactForm');
        const successMsg = document.getElementById('successMsg');

        // Helper to show/hide error messages
        const toggleError = (fieldId, show) => {
            const err = document.getElementById('err-' + fieldId);
            err.style.display = show ? 'block' : 'none';
        };

        // Basic validation logic
        const validate = () => {
            let valid = true;

            // Name
            const name = form.name.value.trim();
            if (!name) { toggleError('name', true); valid = false; }
            else { toggleError('name', false); }

            // Email
            const email = form.email.value.trim();
            const emailPattern = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
            if (!emailPattern.test(email)) { toggleError('email', true); valid = false; }
            else { toggleError('email', false); }

            // Phone (optional)
            const phone = form.phone.value.trim();
            if (phone) {
                const phonePattern = /^\\+?[0-9\\s\\-]{7,15}$/;
                if (!phonePattern.test(phone)) { toggleError('phone', true); valid = false; }
                else { toggleError('phone', false); }
            } else {
                toggleError('phone', false);
            }

            // Message
            const message = form.message.value.trim();
            if (!message) { toggleError('message', true); valid = false; }
            else { toggleError('message', false); }

            return valid;
        };

        // Simulated submit (replace with real fetch/AJAX in production)
        form.addEventListener('submit', e => {
            e.preventDefault();               // stop real submission
            successMsg.style.display = 'none'; // hide previous success

            if (validate()) {
                // Simulate network latency
                setTimeout(() => {
                    form.reset();               // clear fields
                    successMsg.style.display = 'block';
                }, 600);
            } else {
                // Focus first invalid field for better UX
                const firstError = document.querySelector('.error-msg[style*="block"]');
                if (firstError) {
                    const inputId = firstError.id.replace('err-','');
                    document.getElementById(inputId).focus();
                }
            }
        });
    </script>
</body>
</html>
```

---

## How the page works

| Section | What it does | Why it matters |
|---------|--------------|----------------|
| **HTML5 semantic markup** (`<section>`, `<form>`, `<label>`) | Improves SEO and accessibility | Screen readers and search engines understand the structure better |
| **ARIA & required attributes** (`aria-required`, `aria-live`) | Announces errors/success to assistive tech | Users with visual impairments get feedback |
| **Responsive layout** (`max-width`, media query) | Looks good on phones, tablets, and desktops | No extra CSS frameworks needed |
| **CSS reset & simple styling** | Consistent look across browsers | Keeps the code lightweight |
| **Clientâside validation** (JS) | Checks required fields, email format, optional phone pattern | Prevents obvious mistakes before the server sees the data |
| **Success message** (`role="status"`) | Shows a friendly confirmation after âsubmissionâ | Gives users clear feedback that their data was accepted |
| **No external dependencies** | One file, zero CDN calls | Faster load, works offline, easy to embed in any project |

---

### Customising the form

| What you might want to change | Where to edit |
|------------------------------|---------------|
| **Add more fields** (e.g., address, file upload) | Duplicate a `<div class="form-group">` block and adjust the `id`/`name` attributes. Add a corresponding error `<p>` and validation rule in the script. |
| **Change colors / fonts** | Edit the CSS variables at the top (`background`, `color`, `.btn-primary` background, etc.). |
| **Connect to a real backend** | Replace the `setTimeout` block inside the `submit` handler with a `fetch('/your-endpoint', {method:'POST', body: new FormData(form)})` call. Handle the response and display success/error accordingly. |
| **Use a CSS framework** | Remove the `<style>` block and add a link to Tailwind/Bootstrap. Then replace class names (`form-group`, `btn-primary`, etc.) with the frameworkâs utilities. |

---

### Accessibility checklist (already covered)

- **Labels are explicitly bound** to inputs via `for`/`id`.  
- **Required fields** are marked with `aria-required="true"` and a visual asterisk.  
- **Error messages** are placed directly after the related input and toggled with `display:block` so screen readers announce them.  
- **Success message** uses `role="status"` and `aria-live="polite"` for polite announcements.  
- **Keyboard navigation** works because no custom focus traps are introduced.  

Feel free to copyâpaste the code into a file called `contact.html`, open it in a browser, and start tweaking! ð
