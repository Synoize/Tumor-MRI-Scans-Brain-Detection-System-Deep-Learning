import React, { useState } from "react";
import { ChevronUp, ChevronDown, MoveRight } from 'lucide-react';
import { useAppContext } from "../../context/AppContext";

const FAQ = () => {
  const { navigate } = useAppContext();

  const faqs = [
    {
      question: "What is the main objective of this project?",
      answer: `The main goal of this project is to automatically detect and classify brain tumors from MRI images using deep learning. 
      By leveraging Convolutional Neural Networks (CNNs) and transfer learning (VGG16), the system can accurately predict whether a brain MRI scan contains a tumor and identify its type (e.g., Glioma, Meningioma, Pituitary, or No Tumor).`,
    },
    {
      question: "Which technologies and libraries are used in this project?",
      answer: `The project uses Python as the main programming language with the following frameworks and libraries:`,
      list: [
        "TensorFlow / Keras: Deep learning model building",
        "NumPy & Pandas: Data processing",
        "PIL: Image preprocessing",
        "Matplotlib: Visualization",
        "Scikit-learn: Evaluation metrics and confusion matrix",
        "Google Colab: Training environment with GPU support",
      ],
    },
    {
      question: "What dataset was used to train the model?",
      answer: `We used the Brain MRI Dataset available on Kaggle, which contains labeled MRI images under four categories:`,
      list: [
        "Glioma Tumor",
        "Meningioma Tumor",
        "Pituitary Tumor",
        "No Tumor",
      ],
    },
    {
      question: "How does the model work?",
      answer: "The system follows these steps:",
      list: [
        "Data Preprocessing: MRI images are resized, normalized, and augmented.",
        "Feature Extraction: The pre-trained VGG16 model extracts key visual patterns from images.",
        "Classification: Fully connected layers analyze these patterns to classify tumor types.",
        "Prediction: The model predicts the tumor class with a confidence score.",
      ],
    },
    {
      question: "What is Transfer Learning, and why is it used here?",
      answer: `Transfer Learning allows us to use a pre-trained model (like VGG16 trained on ImageNet) and fine-tune it for a specific task — in this case, brain tumor classification. 
      This saves training time, requires fewer data, and improves accuracy since the base model already understands fundamental image features.`,
      highlight: true,
    },
  ];

  const [openIndex, setOpenIndex] = useState(null);

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section className="container mx-auto mb-20 px-4 md:px-12 text-center space-y-12 text-black">
      <div>
        <h3 className="text-sm font-bold uppercase tracking-wide text-special-red">
          Frequently Asked Questions
        </h3>
        <h1 className="text-2xl md:text-5xl max-w-4xl mx-auto font-light mt-4 px-4 leading-snug">
          Commonly asked questions about the model — project’s purpose, working,
          tech stack, and applications.
        </h1>
      </div>

      {/* FAQ Items */}
      <div className="max-w-2xl mx-auto p-4 md:p-0 space-y-4 text-left">
        {faqs.map((faq, index) => (
          <div
            key={index}
            onClick={() => toggleFAQ(index)}
            className={`faq-item font-light hover:border border-gray-200 md:text-lg cursor-pointer transition-all rounded-md p-4 ${faq.highlight ? "bg-gray-400" : ""
              }`}
          >
            <div className="flex justify-between items-center">
              <h3 className="w-5/6 font-normal">{faq.question}</h3>
              <button className="toggle bg-special-red/10 w-8 h-8 flex items-center justify-center rounded-full">
                {openIndex === index ? <ChevronDown className="text-special-red" size={22}/> : <ChevronUp className="text-special-red" size={22}/>}
              </button>
            </div>

            {/* Answer Section */}
            {openIndex === index && (
              <div className="mt-3 space-y-2 animate-fadeIn">
                <p>{faq.answer}</p>
                {faq.list && (
                  <ul className="list-disc list-inside space-y-1">
                    {faq.list.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* More FAQs Button */}
      <div onClick={() => { navigate('/faqs'); scrollTo(0, 0) }}
        className="inline-flex items-center justify-center px-6 py-3 bg-special-red/10 text-black rounded-md hover:bg-special-red/15 shadow-sm transition group cursor-pointer"
      >
        More FAQ's
        <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x[10px] transition-all duration-500">
          <MoveRight size={20} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
        </span>
      </div>
    </section>
  );
};

export default FAQ;
