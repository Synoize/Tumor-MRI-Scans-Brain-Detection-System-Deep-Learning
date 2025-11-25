import React from "react";
import { assets } from "../../assets/assets";
import { Check } from "lucide-react";

const Solutions = () => {
  return (
    <section className="flex flex-col md:flex-row items-start justify-between bg-special-green container mx-auto mb-20">
      {/* Left Text Section */}
      <div className="w-full h-full md:w-1/2 p-12 space-y-12 text-start font-light text-black">
        <div>
          <h2 className="text-2xl md:text-5xl leading-snug">
            We provide comprehensive <br className="hidden md:block" /> medical
            solutions
          </h2>
          <p className="md:text-xl mt-6">
            Brain tumors are life-threatening abnormalities caused by
            uncontrolled growth of brain cells. Traditional diagnosis using MRI
            scans requires expert medical interpretation, which is time-consuming
            and prone to human error.
          </p>
        </div>

        <div>
          <h3 className="text-2xl md:text-5xl mb-6">Our Value Proposition</h3>
          <ul className="space-y-4 md:text-xl">
            <li className="flex items-center gap-3 text-start">
              <Check className="text-special-red" size={20} />
              High accuracy with minimal preprocessing.
            </li>
            <li className="flex items-center gap-3 text-start">
              <Check className="text-special-red" size={20} />
              Automated tumor detection reduces diagnostic time.
            </li>
            <li className="flex items-center gap-3 text-start">
              <Check className="text-special-red" size={20} />
              Can be easily extended to multi-class classification.
            </li>
          </ul>
        </div>
      </div>

      {/* Right Image Section */}
      <div className="w-full md:w-1/2 h-full">
        <img
          loading="lazy"
          src={assets.solutions}
          alt="Solutions"
          className="w-full h-full object-cover"
        />
      </div>
    </section>
  );
};

export default Solutions;
