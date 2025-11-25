import { Github, Instagram, Linkedin, Twitter, Youtube } from "lucide-react";
import React from "react";
import { assets } from "../assets/assets";
import { Link } from "react-router-dom";

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="container mx-auto border-t border-gray-200">
            <div className="w-full p-8 md:p-12 flex flex-col md:flex-row justify-between items-start gap-10 text-gray-800">
                {/* Left Section */}
                <div className="space-y-12 max-w-xl">
                    <div className="space-y-3">
                        <Link to="/" className="flex items-center space-x-2">
                            <img src={assets.logo} alt="tumor" className="h-12 w-auto" />
                        </Link>
                        <h2 className="text-2xl font-light leading-snug">
                            Tumor the AI-powered System
                            <br />
                            designed to assist radiologists and medical professionals in
                            achieving faster, more reliable diagnoses, reducing human error, and improving patient outcomes.
                        </h2>
                    </div>
                    <div className="space-y-3">
                        <div className="font-medium leading-snug">
                            <h2>Address:</h2>
                            <p className="font-light">Saharsa, Bihar, India</p>
                        </div>
                        <div className="font-medium leading-snug">
                            <h2>Email:</h2>
                            <p className="font-light">shivamsingh.dev0@gmail.com</p>
                        </div>
                    </div>
                </div>

                {/* Center Navigation */}
                <div className="flex flex-wrap gap-x-10 gap-y-2 text-xl font-light">
                    <div className="flex flex-col space-y-1">
                        <a href="/" className="hover:underline">Home</a>
                        <a href="./pages/about.html" className="hover:underline">About</a>
                        <a href="./pages/tools.html" className="hover:underline">Tools</a>
                        <a href="./pages/case-studies.html" className="hover:underline">Case Studies</a>
                        <a href="./pages/resources.html" className="hover:underline">Resources</a>
                    </div>
                    <div className="flex flex-col space-y-1">
                        <a href="./pages/blogs.html" className="hover:underline">Blogs</a>
                        <a href="./pages/subscribe.html" className="hover:underline">Subscribe</a>
                    </div>
                </div>

                {/* Social Links */}
                <div className="space-y-4 text-2xl font-light leading-snug">
                    <h4>Follow Us</h4>
                    <div className="flex gap-6 text-gray-600 text-xl">
                        <a href="https://www.github.com/synoize" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="transform transition-transform duration-300 ease-in-out hover:scale-110">
                            <Github size={24} />
                        </a>
                        <a href="https://www.linkedin.com/in/synoize" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="transform transition-transform duration-300 ease-in-out hover:scale-110">
                            <Linkedin size={24} />
                        </a>
                        <a href="https://x.com/synoize" target="_blank" rel="noopener noreferrer" aria-label="X / Twitter" className="transform transition-transform duration-300 ease-in-out hover:scale-110">
                            <Twitter size={24} />
                        </a>
                        <a href="https://instagram.com/synoize" target="_blank" rel="noopener noreferrer" aria-label="Instagram" className="transform transition-transform duration-300 ease-in-out hover:scale-110">
                            <Instagram size={24} />
                        </a>
                        <a href="https://youtube.com/synoize" target="_blank" rel="noopener noreferrer" aria-label="YouTube" className="transform transition-transform duration-300 ease-in-out hover:scale-110">
                            <Youtube size={24} />
                        </a>
                    </div>
                </div>
            </div>

            {/* Bottom Bar */}
            <div className="border-t border-gray-100 py-6 text-center text-sm text-gray-500 space-y-1">
                <p>Copyright © {currentYear} tumer – All Rights Reserved.</p>
                <p>
                    Developed by{" "}
                    <a href="https://synoize.vercel.app" target="_blank" rel="noopener noreferrer" className="font-medium text-gray-700 hover:text-special-red">
                        Team #KS
                    </a>
                </p>
            </div>
        </footer>
    );
};

export default Footer;
