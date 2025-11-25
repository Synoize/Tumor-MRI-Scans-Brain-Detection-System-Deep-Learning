import logo from "./logo.png";
import scan from "./scan.png";
import solutions from "./solutions.png";
import about from "./about.png";
import case_studies from "./case_studies.png";
import donation from "./donation.png";
import resources from "./resources.png";
import top_banner_1 from "./top_banner_1.png";
import top_banner_2 from "./top_banner_2.png";
import top_banner_3 from "./top_banner_3.png";
import healthcare from "./healthcare.png";
import telemedicine from "./telemedicine.png";
import research from "./research.png";
import remove_icon from "./remove_icon.svg";
import dashboard from "./dashboard.svg";
import document_list from "./document_list.svg";
import admin_list from "./admin_list.svg";
import user_list from "./user_list.svg";
import contact_list from "./contact_list.svg"
import upload_area from "./upload_area.png";
import profile_icon from "./profile_icon.png";
import menu_icon from "./menu_icon.svg";
import contact from "./contact.png";


export const assets = {
  logo,
  scan,
  solutions,
  about,
  case_studies,
  resources,
  donation,
  remove_icon,
  dashboard,
  document_list,
  user_list,
  admin_list,
  contact_list,
  upload_area,
  profile_icon,
  menu_icon,
  contact,
};

export const tools = [
  {
    image: scan,
    title: "Brain Tumor Detection",
    description:
      "Upload MRI scans to detect and classify brain tumors using our advanced deep learning model.",
    link: "brain-tumor-detection",
  },
];

export const sliderImages = [
  top_banner_1,
  top_banner_2,
  top_banner_3,
];

export const blogs = [
  {
    title: "Understanding Brain Tumor Detection Using Deep Learning",
    date: "March 15, 2025",
    description:
      "The integration of deep learning with medical imaging has revolutionized early detection systems. This article explains how CNNs analyze MRI scans to detect brain tumors with high precision.",
    link: "./pages/news/news3.html",
  },
  {
    title: "Building Our Brain Tumor Detection Model",
    date: "March 25, 2025",
    description:
      "This blog dives into the technical foundation of our system.",
    link: "./pages/news/news3.html",
  },
  {
    title: "How Artificial Intelligence is Revolutionizing Medical Diagnosis",
    date: "April 2, 2025",
    description:
      "This article explores the global impact of AI in healthcare.",
    link: "./pages/news/news3.html",
  },
  {
    title: "From MRI Image to Diagnosis",
    date: "April 10, 2025",
    description:
      "This post follows a real-world case study using our model.",
    link: "./pages/news/news3.html",
  },
];

export const industries = [
  {
    img: healthcare,
    title: "Healthcare & Hospitals",
    description:
      "Hospitals and diagnostic centers can integrate this system into their radiology departments for real-time brain tumor detection.",
    helps: [
      "Automates MRI scan analysis.",
      "Reduces manual workload for radiologists.",
      "Helps in early tumor diagnosis and treatment planning.",
      "Improves overall patient outcomes and reduces diagnostic delays.",
    ],
  },
  {
    img: research,
    title: "Medical Research Institute",
    description:
      "Research organizations can use this project to study brain tumor patterns, image classification, and AI-driven diagnostics.",
    helps: [
      "Enables dataset collection and pattern recognition.",
      "Supports research on tumor behavior and model accuracy.",
      "Can be customized for clinical trials or AI healthcare research.",
    ],
  },
  {
    img: telemedicine,
    title: "Telemedicine Platforms",
    description:
      "In remote or underdeveloped areas, telemedicine services can use this AI model for remote MRI analysis.",
    helps: [
      "Provides instant diagnostic reports to doctors in different locations.",
      "Reduces the dependency on local specialists.",
      "Makes advanced diagnostics accessible to rural populations.",
    ],
  },
];
