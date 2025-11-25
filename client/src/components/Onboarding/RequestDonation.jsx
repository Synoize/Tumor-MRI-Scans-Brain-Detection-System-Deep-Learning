import React, { useState } from "react";
import { MoveRight } from 'lucide-react';
import { assets } from "../../assets/assets";
import { useAppContext } from "../../context/AppContext";

const RequestDonation = () => {
    const { navigate } = useAppContext();

  const [email, setEmail] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Subscribed with:", email);
    setEmail("");
  };

  return (
    <section className="container mx-auto relative">
      <div className="flex flex-col md:flex-row items-start justify-center bg-special-red/10">
        {/* Left Content */}
        <div className="md:w-1/2 w-full h-full p-8 md:p-12 space-y-12 text-black">
          <div>
            <h4 className="text-sm font-bold uppercase tracking-wide text-special-red">
              Request for Donation
            </h4>
            <h2 className="text-2xl md:text-5xl font-light leading-snug mt-4">
              Your support can help us:
            </h2>
            <ul className="font-light flex flex-col mt-4 space-y-4 md:text-xl">
              <li>Expand our MRI datasets with diverse and high-quality images.</li>
              <li>Improve model accuracy and performance.</li>
              <li>Develop a user-friendly platform for hospitals and medical professionals.</li>
              <li>Conduct more case studies and validation research.</li>
              <li>Share open-source tools with students, researchers, and healthcare innovators.</li>
            </ul>
          </div>

          <div onClick={() => { navigate('/know-more'); scrollTo(0, 0) }}
            className="inline-flex items-center justify-center px-6 py-3 bg-white text-black rounded-md hover:bg-slate-50 shadow-sm transition mt-8 group cursor-pointer"
          >
            Know More
            <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x[10px] transition-all duration-500">
              <MoveRight size={20} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
            </span>
          </div>
        </div>

        {/* Right Image */}
        <div className="md:w-1/2 w-full bg-special-red">
          <img
            loading="lazy"
            src={assets.donation}
            alt="Donation"
            className="w-full h-auto object-cover"
          />
        </div>
      </div>

      {/* Subscribe Section */}
      <div className="w-full md:relative flex justify-center md:items-end">
        <div className="md:absolute px-8 py-6 w-full md:max-w-6xl mx-auto z-20 bg-white/40 backdrop-blur-xs flex flex-col md:flex-row justify-between items-center gap-6 text-xl rounded-2xl mb-10 text-center">
          <h2>Subscribe</h2>
          <p className="font-light text-lg">
            Receive updates about our research, donor acknowledgments, and project progress.
          </p>
          <form
            onSubmit={handleSubmit}
            className="border border-gray-200 md:border-0 rounded-xl overflow-hidden bg-white text-sm flex justify-center items-center flex-nowrap"
          >
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="p-3 outline-none"
              placeholder="Enter Email"
            />
            <button
              type="submit"
              className="px-6 py-3 bg-special-green hover:bg-green-200 cursor-pointer"
            >
              SUBSCRIBE
            </button>
          </form>
        </div>
      </div>
    </section>
  );
};

export default RequestDonation;
